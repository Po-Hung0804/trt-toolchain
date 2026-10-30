"""
Usage (inside the container, from /workspace/proj):
    python scripts/bench_trt.py --engine models/yolo11n_fp16.engine --precision fp16
    python scripts/bench_trt.py --engine models/yolo11n_tf32.engine --precision tf32 --batches 1 4 8

"""

import argparse
from datetime import datetime
from pathlib import Path

import tensorrt as trt
import torch 

from bench import append_csv, gpu_telemetry, summarize, time_gpu

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}

def load_engine(path):
    logger=trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None, f"failed to load {path} (built with another TensorRT version or GPU?)"
    return engine

def io_names(engine):
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
    assert len(inputs) == 1 and len(outputs) == 1, f"expected 1 input / 1 output, got {inputs} / {outputs}"
    return inputs[0], outputs[0]


def bench_batch(engine, ctx, in_name, out_name, batch, imgsz, warmup, iters):
    ctx.set_input_shape(in_name, (batch, 3, imgsz, imgsz))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
 
    in_dtype = TRT_TO_TORCH[engine.get_tensor_dtype(in_name)]
    out_dtype = TRT_TO_TORCH[engine.get_tensor_dtype(out_name)]
 
    d_in = torch.randn(batch, 3, imgsz, imgsz, device="cuda").to(in_dtype)
    d_out = torch.empty(out_shape, device="cuda", dtype=out_dtype)
    h_in = torch.empty(d_in.shape, dtype=in_dtype, pin_memory=True).copy_(d_in.cpu())
    h_out = torch.empty(out_shape, dtype=out_dtype, pin_memory=True)
 
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
 
    # A dedicated stream: enqueueV3 on the default stream adds extra synchronization.
    stream = torch.cuda.Stream()
    handle = stream.cuda_stream
 
    def pure():
        ctx.execute_async_v3(handle)
 
    def io():
        d_in.copy_(h_in, non_blocking=True)
        ctx.execute_async_v3(handle)
        h_out.copy_(d_out, non_blocking=True)
 
    with torch.cuda.stream(stream):  # CUDA events in time_gpu record on this stream
        t_pure = time_gpu(pure, warmup, iters)
        t_io = time_gpu(io, warmup, iters)
    return t_pure, t_io

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--precision", required=True, help="label for the CSV, e.g. tf32 / fp16 / int8")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--out", default="results/bench_tensorrt.csv")
    args = p.parse_args()
 
    assert torch.cuda.is_available(), "CUDA not available: did you run the container with --gpus all?"
 
    path = Path(args.engine)
    engine = load_engine(path)
    in_name, out_name = io_names(engine)
    _, _, max_shape = engine.get_tensor_profile_shape(in_name, 0)
    ctx = engine.create_execution_context()
 
    act_bytes = getattr(engine, "device_memory_size_v2", None) or engine.device_memory_size
    common = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": path.stem,
        "backend": "tensorrt",
        "precision": args.precision,
        "imgsz": args.imgsz,
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "engine_layers": engine.num_layers,
        "engine_mb": round(path.stat().st_size / 2**20, 1),
        "activation_mb": round(act_bytes / 2**20, 1),
    }
    out = Path(args.out)
    rows = []
 
    for batch in args.batches:
        if batch > max_shape[0]:
            print(f"skip batch {batch}: engine profile max is {max_shape[0]}")
            continue
        tele = gpu_telemetry()
        t_pure, t_io = bench_batch(engine, ctx, in_name, out_name, batch, args.imgsz, args.warmup, args.iters)
        for mode, times in (("pure", t_pure), ("io", t_io)):
            stats = summarize(times)
            row = {
                **common,
                "mode": mode,
                "batch": batch,
                **stats,
                "fps": round(batch * 1000.0 / stats["mean_ms"], 1),
                **tele,
            }
            append_csv(out, row)
            rows.append(row)
 
    print(f"\n{path.name}: {engine.num_layers} layers, engine {common['engine_mb']} MB, "
          f"activations {common['activation_mb']} MB")
    print(f"{'mode':<5} {'batch':>5} {'p50':>8} {'p99':>8} {'max':>8} {'fps':>8} {'temp':>5}")
    for r in rows:
        print(
            f"{r['mode']:<5} {r['batch']:>5} {r['p50_ms']:>8} {r['p99_ms']:>8} "
            f"{r['max_ms']:>8} {r['fps']:>8} {str(r['gpu_temp_c']):>5}"
        )
    print(f"\nAppended {len(rows)} rows to {out}")
 
 
if __name__ == "__main__":
    main()