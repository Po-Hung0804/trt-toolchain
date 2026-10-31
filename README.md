# trt-toolchain

Takes a PyTorch vision model to an optimized TensorRT engine and measures what each step gains in latency and costs in accuracy. Runs on a laptop RTX 5050 (8 GB) in WSL2 + Docker, as a stand-in for edge GPUs such as Jetson.

```
PyTorch ──► ONNX (verified vs. PyTorch) ──► TensorRT engine (TF32 / FP16) ──► latency + mAP
```

## Results: YOLO11n, COCO val2017

| Variant | mAP50-95 | Batch 1 p50 / p99 (ms) | Batch 8 p50 (ms) | Batch 8 FPS |
|---|---|---|---|---|
| PyTorch (TF32 conv) | 0.388 | 4.99 / 13.52 | 15.93 | 501 |
| TensorRT TF32 | 0.388 | 2.47 / 4.05 | 10.05 | 793 |
| TensorRT FP16 | 0.388 | 2.09 / 3.70 | 5.24 | 1,521 |

| Speedup (p50) | TensorRT optimization | FP16 | Total |
|---|---|---|---|
| Batch 1 | 2.02× | 1.18× | 2.38× |
| Batch 8 | 1.58× | 1.92× | 3.04× |

- **The bottleneck moves with batch size.** At batch 1 the GPU is underused, so most of the gain comes from TensorRT fusing layers and cutting kernel launches. At batch 8 the GPU is compute-bound and FP16 contributes almost 2×.
- **p99 improves more than p50.** TensorRT alone cuts batch-1 p99 from 13.5 to 4.0 ms by removing per-layer Python dispatch, which matters for frame deadlines.
- **No accuracy loss** from TensorRT or FP16 (within ±0.0002 mAP).

Method: 640×640 input, 100 warm-up + 1,000 timed runs, CUDA events, model forward only. PyTorch and TensorRT both use TF32 for convolutions by default, so PyTorch → TensorRT TF32 isolates the engine's effect. Accuracy uses square 640×640 input for every backend.

## Usage

Needs WSL2, Docker and the NVIDIA Container Toolkit. Container: `nvcr.io/nvidia/pytorch:26.02-py3` (PyTorch 2.11, CUDA 13.1, TensorRT 10.15).

```bash
docker build -t trt-toolchain:dev .

./run.sh python scripts/pipeline.py                    # whole pipeline + summary table
```

Or run each step on its own:

```bash
./run.sh python scripts/bench.py                       # PyTorch latency
./run.sh python scripts/eval.py                        # PyTorch mAP (needs COCO in data/coco)
./run.sh python scripts/export_onnx.py                 # ONNX export + numerical check
./run.sh python scripts/build_engine.py --onnx models/yolo11n.onnx --precision fp16
./run.sh python scripts/bench_trt.py --engine models/yolo11n_fp16.engine --precision fp16
./run.sh python scripts/eval.py --model models/yolo11n_fp16.engine --precision fp16
```

Results go to `results/*.csv`, and `pipeline.py` writes a comparison table to `results/summary_*.md`. Each export and build writes a JSON record to `models/`.

## Roadmap

- [x] PyTorch baseline, ONNX export, TensorRT TF32 / FP16
- [x] Single command for the whole pipeline
- [ ] INT8 with calibration and per-layer mixed precision
- [ ] A Transformer model (Depth Anything V2 Small)
- [ ] Nsight profiling, GPU pre-processing, CUDA Graphs

Uses [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and [COCO](https://cocodataset.org).
