"""
Run the whole toolchain with one command 
Pytorch baseline (latency + mAP)
Each steps calls the existing script, so they still work on their own. 
"""

import argparse
import subprocess
import sys
import time 

from datetime import datetime
from pathlib import Path 

SCRIPTS= Path(__file__).resolve().parent

class Runner:
    def __init__(self, dry_run):
        self.dry_run=dry_run
        # (name, seconds)
        self.steps=[]

    def run(self, name, script, *args):
        cmd = [sys.executable, str(SCRIPTS/script), *map(str, args)]
        print(f"\n=== {name} ===\n$ {' '.join(cmd[1:])}", flush=True) 
        if self.dry_run:
            return 
        t0= time.perf_counter()
        result=subprocess.run(cmd)
        secs=time.perf_counter() - t0
        self.steps.append((name, secs))
        if result.returncode != 0:
            sys.exit(f"\nstep failed: {name} (exit code {result.returncode})")

    def cool(self, seconds):
        if seconds > 0 and not self.dry_run:
            print(f"\ncooling down {seconds}s ...", flush=True)
            time.sleep(seconds)

def summarize(run_start, stem, batches):
    # build a markdown table from the rows this run append to the csvs
    import pandas as pd

    def load(path):
        p=Path(path)
        if not p.exists():
            return pd.DataFrame()
        df= pd.read_csv(p)
        return df[df["timestamp"] >= run_start]

    lat = pd.concat([load("results/baseline_pytorch.csv"), load("results/bench_tensorrt.csv")], ignore_index=True)
    acc = load("results/baseline_accuracy.csv")
    if lat.empty:
        return "no benchmark rows found for this run"
    lat = lat[lat["mode"] == "pure"]
 
    b_lo, b_hi = min(batches), max(batches)
    variants = list(dict.fromkeys(zip(lat["backend"], lat["precision"])))
    base = None
    rows = []
    for backend, precision in variants:
        v = lat[(lat["backend"] == backend) & (lat["precision"] == precision)]
        lo = v[v["batch"] == b_lo].iloc[-1]
        hi = v[v["batch"] == b_hi].iloc[-1]
        a = acc[(acc["backend"] == backend) & (acc["precision"] == precision)] if not acc.empty else acc
        m = f"{a.iloc[-1]['map50_95']:.4f}" if not a.empty else "–"
        if base is None:
            base = (lo["p50_ms"], hi["p50_ms"])
        name = "PyTorch" if backend == "pytorch" else f"TensorRT {precision.upper()}"
        rows.append(
            f"| {name} | {m} | {lo['p50_ms']:.2f} / {lo['p99_ms']:.2f} | {hi['p50_ms']:.2f} | "
            f"{hi['fps']:,.0f} | {base[0] / lo['p50_ms']:.2f}× | {base[1] / hi['p50_ms']:.2f}× |"
        )
 
    header = (
        f"## {stem} — {run_start}\n\n"
        f"| Variant | mAP50-95 | Batch {b_lo} p50 / p99 (ms) | Batch {b_hi} p50 (ms) | Batch {b_hi} FPS "
        f"| Speedup b{b_lo} | Speedup b{b_hi} |\n|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows) + "\n"

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--precisions", nargs="+", default=["tf32", "fp16"], choices=["fp32", "tf32", "fp16"])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--subset", type=int, default=0, help="images for mAP, 0 = full val2017")
    p.add_argument("--cooldown", type=int, default=30, help="seconds between benchmarks")
    p.add_argument("--skip-baseline", action="store_true", help="skip the PyTorch latency and mAP steps")
    p.add_argument("--skip-eval", action="store_true", help="skip all mAP steps")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
 
    run_start = datetime.now().isoformat(timespec="seconds")
    stem = Path(args.model).stem
    onnx = Path("models") / f"{stem}.onnx"
    batches = sorted(set(args.batches))
    profile = (batches[0], batches[len(batches) // 2], batches[-1])  # min, opt, max
    max_eval_batch = profile[2]
    r = Runner(args.dry_run)
 
    if not args.skip_baseline:
        r.run("PyTorch latency", "bench.py", "--model", args.model, "--batches", *batches,
              "--iters", args.iters, "--no-e2e")
        if not args.skip_eval:
            r.run("PyTorch mAP", "eval.py", "--model", args.model, "--subset", args.subset,
                  "--batch", max_eval_batch)
        r.cool(args.cooldown)
 
    r.run("ONNX export", "export_onnx.py", "--model", args.model)
 
    for prec in args.precisions:
        engine = Path("models") / f"{stem}_{prec}.engine"
        r.run(f"build {prec}", "build_engine.py", "--onnx", onnx, "--precision", prec, "--batch", *profile)
        r.cool(args.cooldown)
        r.run(f"TensorRT {prec} latency", "bench_trt.py", "--engine", engine, "--precision", prec,
              "--batches", *batches, "--iters", args.iters)
        if not args.skip_eval:
            r.run(f"TensorRT {prec} mAP", "eval.py", "--model", engine, "--precision", prec,
                  "--subset", args.subset, "--batch", max_eval_batch)
 
    if args.dry_run:
        return
 
    table = summarize(run_start, stem, batches)
    out = Path("results") / f"summary_{stem}_{run_start.replace(':', '')}.md"
    out.write_text(table)
 
    print("\n" + "=" * 60)
    print(table)
    print("step times: " + ", ".join(f"{n} {s:.0f}s" for n, s in r.steps))
    print(f"summary -> {out}")
 
 
if __name__ == "__main__":
    main()