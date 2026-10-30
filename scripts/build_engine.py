"""
Build a TensorRT engine from ONNX with the Python API

Precision 
    fp32 strict FP32 (TF32 disable)
    tf32 TensorRT defalut: FP32 math, TF32 allowed for conv/matmul
    fp16 FP16 allowed; TensorRT picks FP16 or FP32 per layer, whichever is faster

Output:
    <name>.engine        the serialized engine
    <name>.json          build settings, versions, build time, layer count
    <name>.layers.json   per-layer info: which ONNX ops were fused into each engine layer
A timing cache (models/timing.cache) is reused across builds, so rebuilds are much faster.
"""

import argparse
import json
import time 

from pathlib import Path
import tensorrt as trt

LOGGER=trt.Logger(trt.Logger.WARNING)

def parse_onnx(builder, onnx_path):
    network=builder.create_network(0)
    parser=trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")
    return network

def set_precision(config, precision):
    # TF32 is default in TensorRT; only strict fp32 turns it off
    if precision=='fp32':
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "tf32":
        raise ValueError(f"unsupported precision: {precision}")

def add_profile(builder, config, network, imgsz, batch):
    # one optimization profile over the dynamic batch axis
    assert network.num_inputs == 1, "expected a single input tensor"
    inp = network.get_input(0)
    bmin, bopt, bmax = batch
    shape = lambda b: (b, 3, imgsz, imgsz)  # noqa: E731
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, shape(bmin), shape(bopt), shape(bmax))
    config.add_optimization_profile(profile)
    return inp.name

def load_timing_cache(config, path: Path):
    blob = path.read_bytes() if path.exists() else b""
    cache = config.create_timing_cache(blob)
    config.set_timing_cache(cache, ignore_mismatch=False)
 
 
def save_timing_cache(config, path: Path):
    path.write_bytes(memoryview(config.get_timing_cache().serialize()))
 
 
def layer_info(engine_bytes: bytes) -> dict:
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    return {"num_layers": engine.num_layers, "info": info}
 
 
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--precision", choices=["fp32", "tf32", "fp16"], default="fp16")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, nargs=3, default=[1, 4, 8], metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--workspace-gb", type=float, default=2.0, help="scratch memory TensorRT may use while building")
    p.add_argument("--timing-cache", default="models/timing.cache")
    p.add_argument("--out", help="engine path (default: models/<onnx stem>_<precision>.engine)")
    args = p.parse_args()
 
    onnx_path = Path(args.onnx)
    out = Path(args.out) if args.out else onnx_path.with_name(f"{onnx_path.stem}_{args.precision}.engine")
    out.parent.mkdir(parents=True, exist_ok=True)
 
    builder = trt.Builder(LOGGER)
    network = parse_onnx(builder, onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * 2**30))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # keeps layer names for the inspector
    set_precision(config, args.precision)
    input_name = add_profile(builder, config, network, args.imgsz, tuple(args.batch))
    cache_path = Path(args.timing_cache)
    load_timing_cache(config, cache_path)
 
    print(f"building {out.name}  precision={args.precision}  batch(min,opt,max)={tuple(args.batch)}")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("engine build failed, see TensorRT messages above")
 
    engine_bytes = bytes(serialized)
    out.write_bytes(engine_bytes)
    save_timing_cache(config, cache_path)
 
    layers = layer_info(engine_bytes)
    out.with_suffix(".layers.json").write_text(json.dumps(layers["info"], indent=2))
 
    meta = {
        "engine": str(out),
        "onnx": str(onnx_path),
        "precision": args.precision,
        "imgsz": args.imgsz,
        "input": input_name,
        "batch_profile": {"min": args.batch[0], "opt": args.batch[1], "max": args.batch[2]},
        "workspace_gb": args.workspace_gb,
        "onnx_nodes": network.num_layers,
        "engine_layers": layers["num_layers"],
        "engine_mb": round(len(engine_bytes) / 2**20, 2),
        "build_seconds": round(build_s, 1),
        "tensorrt": trt.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
 
    print(f"done in {build_s:.1f}s  network layers {meta['onnx_nodes']} -> engine layers {meta['engine_layers']}  "
          f"size {meta['engine_mb']} MB")
    print(f"engine -> {out}\nmetadata -> {out.with_suffix('.json')}\nlayer info -> {out.with_suffix('.layers.json')}")
 
 
if __name__ == "__main__":
    main()
