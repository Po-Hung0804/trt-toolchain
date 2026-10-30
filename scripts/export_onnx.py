"""
steps 
    1. export  torch.onnx.export with a dynamic batch axis
    2. slim    onnxslim  folds constants and removes redundant nodes
    3. check   onnx.checker validate the graph
    4. verify  same inputs through PyTorch (GPU) and ONNX Runtime (CPU), compare outputs

Usage (inside the container, from /workspace/proj):
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model yolov11s.pt --opset 17 # change the model
    python scripts/export_onnx.py --dynamo  # try the newer torch.export-based exporter
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

class OutputOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m=model

    def forward(self, x):
        out=self.m(x)
        return out[0] if isinstance(out, (list, tuple)) else out

def load_image(imgsz) -> np.ndarray:
    # a real image, plain-resized: enough for a numerical equivalence check
    import cv2
    from ultralytics.utils import ASSETS

    img=cv2.imread(str(ASSETS/"bus.jpg"))
    img=cv2.resize(img, (imgsz, imgsz))
    img=img[:, :, ::-1].transpose(2, 0, 1)
    return (np.ascontiguousarray(img, dtype=np.float32) / 255.0)[None]

def op_histogram(path: Path)-> Counter:
    import onnx

    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)

""" step """
def export(model, path, imgsz, opset, max_batch, use_dynamo):
    if use_dynamo:
        # torch.export specializes size-1 dims, so trace with batch 2.
        dummy = torch.zeros(2, 3, imgsz, imgsz, device="cuda")
        batch = torch.export.Dim("batch", min=1, max=max_batch)
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            dynamo=True,
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            dynamic_shapes={"x": {0: batch}},
        )
    else:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device="cuda")
        torch.onnx.export(
            model,
            dummy,
            str(path),
            dynamo=False,
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
            do_constant_folding=True,
        )

def slim(path):
    import onnx
    import onnxslim
 
    onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))

def verify(model, path, imgsz, box_tol, cls_tol):
    import onnxruntime as ort
 
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {
        "real image, batch 1": load_image(imgsz),
        "random, batch 4": np.random.default_rng(0).random((4, 3, imgsz, imgsz), dtype=np.float32),
    }
 
    results = []
    for name, x in inputs.items():
        with torch.inference_mode():
            ref = model(torch.from_numpy(x).cuda()).float().cpu().numpy()
        out = sess.run(None, {"images": x})[0]
        assert ref.shape == out.shape, f"{name}: shape mismatch {ref.shape} vs {out.shape}"
 
        box_diff = float(np.abs(ref[:, :4] - out[:, :4]).max())  # pixels
        cls_diff = float(np.abs(ref[:, 4:] - out[:, 4:]).max())  # probabilities
        det_ref = int((ref[:, 4:].max(1) > 0.25).sum())
        det_out = int((out[:, 4:].max(1) > 0.25).sum())
        ok = box_diff < box_tol and cls_diff < cls_tol and det_ref == det_out
        results.append(
            {
                "input": name,
                "shape": list(out.shape),
                "box_max_abs_diff_px": round(box_diff, 6),
                "cls_max_abs_diff": round(cls_diff, 8),
                "detections_pytorch": det_ref,
                "detections_onnx": det_out,
                "pass": ok,
            }
        )
    return results

""" main """

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-batch", type=int, default=16, help="upper bound for the dynamic batch axis (dynamo only)")
    p.add_argument("--dynamo", action="store_true", help="use the torch.export-based exporter")
    p.add_argument("--no-slim", action="store_true")
    p.add_argument("--box-tol", type=float, default=0.05, help="max allowed box difference, pixels")
    p.add_argument("--cls-tol", type=float, default=1e-3, help="max allowed class-score difference")
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()
 
    import onnx
    from ultralytics import YOLO
 
    assert torch.cuda.is_available(), "CUDA not available: did you run the container with --gpus all?"
 
    # PyTorch uses TF32 for convolutions by default on recent GPUs. Turn it off so the
    # reference is true FP32 and the comparison tests the export, not TF32 rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
 
    yolo = YOLO(args.model)
    yolo.fuse()
    net = OutputOnly(yolo.model.to("cuda").float().eval()).eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.model).stem
    path = out_dir / f"{stem}.onnx"
 
    print(f"[1/4] export  -> {path}  (opset {args.opset}, {'dynamo' if args.dynamo else 'torchscript'} exporter)")
    export(net, path, args.imgsz, args.opset, args.max_batch, args.dynamo)
    ops_before = op_histogram(path)
 
    if args.no_slim:
        ops_after = ops_before
        print("[2/4] slim    skipped")
    else:
        slim(path)
        ops_after = op_histogram(path)
        print(f"[2/4] slim    nodes {sum(ops_before.values())} -> {sum(ops_after.values())}")
 
    onnx.checker.check_model(str(path))
    print("[3/4] check   onnx.checker OK")
 
    print("[4/4] verify  PyTorch (GPU, FP32) vs ONNX Runtime (CPU)")
    results = verify(net, path, args.imgsz, args.box_tol, args.cls_tol)
    for r in results:
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"      {flag}  {r['input']:<20} box diff {r['box_max_abs_diff_px']:.5f}px  "
            f"cls diff {r['cls_max_abs_diff']:.2e}  detections {r['detections_pytorch']} vs {r['detections_onnx']}"
        )
 
    meta = {
        "source": args.model,
        "onnx": str(path),
        "size_mb": round(path.stat().st_size / 2**20, 2),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "exporter": "dynamo" if args.dynamo else "torchscript",
        "input": {"name": "images", "shape": ["batch", 3, args.imgsz, args.imgsz]},
        "output": {"name": "output0", "layout": "(batch, 4 box xywh px + classes, anchors)"},
        "torch": torch.__version__,
        "onnx_version": onnx.__version__,
        "nodes_before_slim": sum(ops_before.values()),
        "nodes_after_slim": sum(ops_after.values()),
        "op_types": dict(ops_after.most_common()),
        "verify": results,
    }
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
 
    print(f"\nTop ops: {', '.join(f'{k}×{v}' for k, v in ops_after.most_common(6))}")
    print(f"Size {meta['size_mb']} MB, metadata -> {meta_path}")
 
    if not all(r["pass"] for r in results):
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
        