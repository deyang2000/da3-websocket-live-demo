import os, glob, statistics
import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

local_path = "/mnt/afs/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
example_path = "/mnt/afs/iq1m/outdoor/cmu.west/Images"
images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))
assert len(images) > 0

model = DepthAnything3.from_pretrained(local_path, local_files_only=True).to(device).eval()

BATCH = min(64, len(images))  # 先测 16 张一批
WARMUP = 3
ITERS = 10

USE_AMP = True
AMP_DTYPE = torch.float16

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

def run_batch():
    pred = model.inference(images[:BATCH])
    _ = pred.depth
    return pred

# warmup
with torch.inference_mode():
    for _ in range(WARMUP):
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                run_batch()
        else:
            run_batch()
torch.cuda.synchronize()

# measure
times_ms = []
with torch.inference_mode():
    for _ in range(ITERS):
        starter.record()
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                pred = run_batch()
        else:
            pred = run_batch()
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))

mean_ms = statistics.mean(times_ms)
throughput = (BATCH * 1000.0) / mean_ms
print(f"Batch={BATCH} mean={mean_ms:.2f} ms -> throughput={throughput:.2f} imgs/s")
print("depth shape:", pred.depth.shape)
