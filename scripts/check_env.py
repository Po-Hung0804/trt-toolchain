import torch
import tensorrt as trt

print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("TensorRT:", trt.__version__)

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("FP16 matmul OK:", tuple(y.shape))
