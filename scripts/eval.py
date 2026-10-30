import argparse 
import csv
import random
from datetime import datetime 
from pathlib import Path

import torch
import yaml

def make_subset(img_dir, n, seed ,out_txt):
    # create a fixed random subset of image path 
    imgs=sorted(img_dir.glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    out_txt.write_text("\n".join(str(p) for p in imgs[:n]) + "\n")
    return out_txt

def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt", help=".pt weights or a TensorRT .engine")
    p.add_argument("--precision", default="fp32", help="label for the CSV, e.g. fp32 / tf32 / fp16 / int8")
    p.add_argument("--names-from", default="yolo11n.pt",
                   help="weights to read class names from when --model is an engine without metadata")
    p.add_argument("--coco", default="data/coco", help="COCO root containing images/ and labels/")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="must be within the engine's optimization profile")
    p.add_argument("--subset", type=int, default=0, help="0 = full val2017")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/baseline_accuracy.csv")
    args = p.parse_args()
 
    from ultralytics import YOLO
 
    assert torch.cuda.is_available(), "CUDA not available: did you run the container with --gpus all?"
 
    root = Path(args.coco).resolve()
    img_dir = root / "images" / "val2017"
    assert img_dir.is_dir(), f"{img_dir} not found: download COCO val2017 first"
 
    work = root / "_eval"
    work.mkdir(exist_ok=True)
    if args.subset:
        val = make_subset(img_dir, args.subset, args.seed, work / f"val2017_{args.subset}.txt")
    else:
        val = img_dir
 
    model_path = Path(args.model)
    backend = {".engine": "tensorrt", ".onnx": "onnx"}.get(model_path.suffix, "pytorch")
 
    yolo = YOLO(args.model, task="detect")
    # Engines built by trtexec carry no Ultralytics metadata, so take names from the .pt weights.
    names = yolo.names if backend == "pytorch" else YOLO(args.names_from).names
 
    # Minimal dataset yaml; class names are COCO's 80 classes.
    data_yaml = work / "coco_val.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(val),  # required key, never used for validation
                "val": str(val),
                "names": names,
            },
            sort_keys=False,
        )
    )
 
    m = yolo.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        # Square 640x640 letterbox for every backend. PyTorch would otherwise use
        # rectangular batches (e.g. 640x480), engines cannot, and the mAP gap would
        # come from input shape rather than precision.
        rect=False,
        device=0,
        plots=False,
        verbose=False,
    )
 
    n_images = args.subset or len(list(img_dir.glob("*.jpg")))
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_path.stem,
        "backend": backend,
        "precision": args.precision,
        "imgsz": args.imgsz,
        "rect": False,
        "images": n_images,
        "seed": args.seed if args.subset else "",
        "map50_95": round(float(m.box.map), 4),
        "map50": round(float(m.box.map50), 4),
    }
    out = Path(args.out)
    append_csv(out, row)
 
    print(f"\n{row['model']}  {n_images} images  mAP50-95={row['map50_95']:.4f}  mAP50={row['map50']:.4f}")
    print(f"Appended to {out}")
 
 
if __name__ == "__main__":
    main()
 