import os
import statistics
import torch
from depth_anything_3.api import DepthAnything3

# -----------------------
# config
# -----------------------
device = torch.device("cuda")

local_path = "/mnt/afs/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
assert os.path.exists(local_path), f"Checkpoint path not found: {local_path}"

WARMUP = 30
ITERS = 200

USE_AMP = True
AMP_DTYPE = torch.float16  # or torch.bfloat16

# DA3 core 期望 5D: (B, S, C, H, W)
B, S, C, H, W = 1, 1, 3, 280, 504

# -----------------------
# speed knobs
# -----------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------
# load model
# -----------------------
print("Loading model...")
da3 = DepthAnything3.from_pretrained(local_path, local_files_only=True).to(device)
da3.eval()

core = getattr(da3, "model", None)
if core is None:
    raise RuntimeError("DepthAnything3 instance has no attribute `.model` (unexpected).")

# -----------------------
# dummy input (5D)
# -----------------------
dummy_input = torch.randn(B, S, C, H, W, device=device, dtype=torch.float32)

# -----------------------
# forward wrapper
# -----------------------
def forward_once(x):
    # core(x) should work now that x is 5D
    y = core(x)
    return y

def touch_output(y):
    # 尽量轻触，避免额外同步开销
    if isinstance(y, torch.Tensor):
        _ = y.shape
        return
    if isinstance(y, (list, tuple)) and len(y) > 0:
        t = y[0]
        if isinstance(t, torch.Tensor):
            _ = t.shape
        return
    if isinstance(y, dict) and len(y) > 0:
        t = next(iter(y.values()))
        if isinstance(t, torch.Tensor):
            _ = t.shape
        return

# -----------------------
# warmup
# -----------------------
print(f"Warmup {WARMUP} iters... AMP={USE_AMP} dtype={AMP_DTYPE if USE_AMP else None}")
with torch.inference_mode():
    for _ in range(WARMUP):
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                y = forward_once(dummy_input)
        else:
            y = forward_once(dummy_input)
        touch_output(y)
torch.cuda.synchronize()

# -----------------------
# timing (CUDA events)
# -----------------------
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

times_ms = []
print(f"Benchmark {ITERS} iters...")
with torch.inference_mode():
    for _ in range(ITERS):
        starter.record()
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                y = forward_once(dummy_input)
        else:
            y = forward_once(dummy_input)
        touch_output(y)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))

times_ms_sorted = sorted(times_ms)
mean_ms = statistics.mean(times_ms)
p50 = times_ms_sorted[int(0.50 * (ITERS - 1))]
p90 = times_ms_sorted[int(0.90 * (ITERS - 1))]
p95 = times_ms_sorted[int(0.95 * (ITERS - 1))]
fps = 1000.0 / mean_ms

print(f"[Forward-only] input=({B},{S},{C},{H},{W}) AMP={USE_AMP} dtype={AMP_DTYPE if USE_AMP else None}")
print(f"Latency (ms): mean={mean_ms:.2f}, p50={p50:.2f}, p90={p90:.2f}, p95={p95:.2f}")
print(f"Approx FPS (1/mean): {fps:.2f}")
