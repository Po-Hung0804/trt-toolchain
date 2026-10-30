import argparse
import csv
import time 
from datetime import datetime 
from pathlib import Path

import numpy as np
import torch


def gpu_telemetry() -> dict:
    # GPU temperature and SM clock
    try:
        import pynvml 
        pynvml.nvmlInit()
        h=pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        }
    except Exception:
        return {"gpu_temp_c": None, "sm_clock_mhz": None}

def summarize(times_ms) -> dict:
    a=np.asarray(times_ms)
    return {
        "mean_ms": round(float(a.mean()), 3),
        "std_ms": round(float(a.std()), 3),
        "min_ms": round(float(a.min()), 3),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "max_ms": round(float(a.max()), 3),
    }

def time_gpu(fn, warmup, iters) -> list[float]:
    # Time a GPU-only callable with CUDA events
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times

def time_wall(fn, warmup, iters) -> list[float]:
    # wall clock timing for work mixed CPU and GPU 
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
 
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times

def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def bench_pure(net, batch, imgsz, warmup, iters):
    x=torch.randn(batch, 3, imgsz, imgsz, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        times = time_gpu(lambda: net(x), warmup, iters)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    return times, peak_mb

def bench_e2e(yolo, imgsz, warmup, iters):
    import cv2
    from ultralytics.utils import ASSETS
 
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    torch.cuda.reset_peak_memory_stats()
    times = time_wall(
        lambda: yolo.predict(img, imgsz=imgsz, device=0, verbose=False),
        warmup,
        iters,
    )
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    return times, peak_mb

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics weights name or path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--no-e2e", action="store_true", help="skip the end-to-end predict() benchmark")
    p.add_argument("--out", default="results/baseline_pytorch.csv")
    args = p.parse_args()
 
    from ultralytics import YOLO
 
    assert torch.cuda.is_available(), "CUDA not available: did you run the container with --gpus all?"
    torch.backends.cudnn.benchmark = True  # let cuDNN pick fastest kernels, fairer vs. TensorRT
 
    yolo = YOLO(args.model)
    yolo.fuse()  # fold Conv+BN, same as Ultralytics does before predict()
    net = yolo.model.to("cuda").eval().float()
 
    common = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": Path(args.model).stem,
        "backend": "pytorch",
        "precision": "fp32",
        "imgsz": args.imgsz,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    out = Path(args.out)
    rows = []
 
    for batch in args.batches:
        tele = gpu_telemetry()
        times, peak_mb = bench_pure(net, batch, args.imgsz, args.warmup, args.iters)
        stats = summarize(times)
        row = {
            **common,
            "mode": "pure",
            "batch": batch,
            **stats,
            "fps": round(batch * 1000.0 / stats["mean_ms"], 1),
            "peak_mem_mb": round(peak_mb, 1),
            **tele,
        }
        append_csv(out, row)
        rows.append(row)
 
    if not args.no_e2e:
        tele = gpu_telemetry()
        times, peak_mb = bench_e2e(yolo, args.imgsz, args.warmup, args.iters)
        stats = summarize(times)
        row = {
            **common,
            "mode": "e2e",
            "batch": 1,
            **stats,
            "fps": round(1000.0 / stats["mean_ms"], 1),
            "peak_mem_mb": round(peak_mb, 1),
            **tele,
        }
        append_csv(out, row)
        rows.append(row)
 
    print(f"\n{'mode':<5} {'batch':>5} {'p50':>8} {'p99':>8} {'max':>8} {'fps':>8} {'mem MB':>8} {'temp':>5}")
    for r in rows:
        print(
            f"{r['mode']:<5} {r['batch']:>5} {r['p50_ms']:>8} {r['p99_ms']:>8} "
            f"{r['max_ms']:>8} {r['fps']:>8} {r['peak_mem_mb']:>8} {str(r['gpu_temp_c']):>5}"
        )
    print(f"\nAppended {len(rows)} rows to {out}")
 
 
if __name__ == "__main__":
    main()