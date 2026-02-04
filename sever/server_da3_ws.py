#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_da3_ws.py

DA3 WebSocket streaming server
- Receive: JPEG bytes (binary WS message)
- Send: colorized depth JPEG bytes (or 16-bit PNG) (binary WS message)

Robustness:
- Per-connection queue (maxsize=1) => drop old frames
- Graceful close: no "Task exception was never retrieved"
"""

import asyncio
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from depth_anything_3.api import DepthAnything3

# -----------------------
# Config
# -----------------------

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8001

    ckpt: str = "/mnt/afs/Depth-Anything-3/checkpoints/DA3NESTED-GIANT-LARGE"

    device: str = "cuda"
    amp_dtype: str = "fp16"   # "fp16" or "bf16"
    tf32: bool = True

    target_h: int = 280
    target_w: int = 504

    # output format
    send_mode: str = "jpeg_color"   # "jpeg_color" or "png16"
    jpeg_quality: int = 70

    # logging
    log_every_n: int = 30  # log every N frames per client


CFG = Config()

def get_amp_dtype(name: str):
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


# -----------------------
# Pre/Post process
# -----------------------

def decode_jpeg_to_bgr(jpg_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr

def preprocess_bgr_to_5d_tensor(bgr: np.ndarray, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    """
    bgr: HxWx3 uint8
    return: (B,S,C,H,W) float32 on GPU, with S=1
    """
    if bgr.shape[0] != target_h or bgr.shape[1] != target_w:
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # HWC uint8 -> CHW float32 [0,1]
    x = torch.from_numpy(rgb).to(device=device, non_blocking=True)
    x = x.permute(2, 0, 1).contiguous()  # (3,H,W)
    x = x.to(torch.float32).div_(255.0)

    # (B,S,C,H,W) with S=1
    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
    return x

def depth_tensor_to_hw_numpy(depth: torch.Tensor) -> np.ndarray:
    """
    depth could be (B,S,H,W) or (B,H,W) or (H,W)
    return HxW float32 numpy
    """
    d = depth
    # peel dims until HxW
    while d.dim() > 2:
        d = d[0]
    return d.detach().float().cpu().numpy()

def depth_to_png16(depth_hw: np.ndarray) -> bytes:
    """
    depth_hw: HxW float32
    -> per-frame normalized uint16 PNG (demo friendly)
    """
    d = np.nan_to_num(depth_hw, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(d.min()), float(d.max())
    if mx - mn < 1e-6:
        u16 = np.zeros_like(d, dtype=np.uint16)
    else:
        u16 = ((d - mn) / (mx - mn) * 65535.0).astype(np.uint16)

    ok, buf = cv2.imencode(".png", u16)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()

def depth_to_color_jpeg(depth_hw: np.ndarray, quality: int = 70) -> bytes:
    """
    depth_hw: HxW float32
    -> color mapped JPEG bytes
    """
    d = np.nan_to_num(depth_hw, nan=0.0, posinf=0.0, neginf=0.0)
    d_min, d_max = float(d.min()), float(d.max())

    if d_max - d_min > 1e-5:
        depth_norm = (255.0 * (d - d_min) / (d_max - d_min)).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(d, dtype=np.uint8)

    color_map = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    ok, buf = cv2.imencode(".jpg", color_map, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# -----------------------
# Server
# -----------------------

class DA3Server:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.amp_dtype = get_amp_dtype(cfg.amp_dtype)

        # speed knobs
        torch.backends.cudnn.benchmark = True
        if cfg.tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.core = None  # assigned after load

    def load(self):
        assert os.path.exists(self.cfg.ckpt), f"Checkpoint not found: {self.cfg.ckpt}"
        print("[DA3] loading:", self.cfg.ckpt)
        da3 = DepthAnything3.from_pretrained(self.cfg.ckpt, local_files_only=True).to(device=self.device).eval()
        self.core = da3.model
        print("[DA3] loaded. core:", type(self.core))

    async def handler(self, ws: websockets.WebSocketServerProtocol):
        """
        Per-connection handler: own queue, receiver task, sender task.
        """
        client = getattr(ws, "remote_address", None)
        print(f"[WS] Client connected: {client}")

        frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

        async def receiver():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        # ignore text frames
                        continue

                    # drop old frame if full
                    if frame_queue.full():
                        try:
                            _ = frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(msg)

            except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
                # client closed
                pass
            except Exception as e:
                print(f"[WS][receiver] error: {repr(e)}")

        async def worker_and_sender():
            frame_count = 0
            t0 = time.time()

            try:
                while True:
                    jpg_bytes = await frame_queue.get()

                    bgr = decode_jpeg_to_bgr(jpg_bytes)
                    if bgr is None:
                        continue

                    x = preprocess_bgr_to_5d_tensor(
                        bgr, self.cfg.target_h, self.cfg.target_w, self.device
                    )

                    with torch.inference_mode(), torch.autocast("cuda", dtype=self.amp_dtype):
                        y = self.core(x)

                    # DA3 output is addict.Dict with keys shown by you
                    # dict_keys(['depth','depth_conf','extrinsics','intrinsics','aux','is_metric','scale_factor'])
                    depth_t = None
                    try:
                        depth_t = y["depth"]
                    except Exception:
                        # fallback if output shape differs
                        if isinstance(y, dict) and "depth" in y:
                            depth_t = y["depth"]

                    if depth_t is None or not isinstance(depth_t, torch.Tensor):
                        # can't extract depth
                        continue

                    depth_hw = depth_tensor_to_hw_numpy(depth_t)

                    if self.cfg.send_mode == "png16":
                        out_bytes = depth_to_png16(depth_hw)
                    else:
                        out_bytes = depth_to_color_jpeg(depth_hw, quality=self.cfg.jpeg_quality)

                    try:
                        await ws.send(out_bytes)
                    except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
                        print("[WS][sender] ws closed, stop sender")
                        break

                    frame_count += 1
                    if self.cfg.log_every_n > 0 and (frame_count % self.cfg.log_every_n == 0):
                        dt = time.time() - t0
                        fps = frame_count / max(dt, 1e-6)
                        print(f"[WS][sender] {client} frames={frame_count} fps={fps:.1f}")

            except asyncio.CancelledError:
                # cancelled by handler
                pass
            except Exception as e:
                print(f"[WS][sender] error: {repr(e)}")

        receiver_task = asyncio.create_task(receiver(), name="receiver")
        sender_task = asyncio.create_task(worker_and_sender(), name="sender")

        done, pending = await asyncio.wait(
            [receiver_task, sender_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # cancel remaining task
        for t in pending:
            t.cancel()
            try:
                await t
            except Exception:
                pass

        print(f"[WS] Client disconnected: {client}")

    async def run(self):
        assert self.core is not None, "Call load() before run()"
        print(f"[WS] Serving on ws://{self.cfg.host}:{self.cfg.port}  send_mode={self.cfg.send_mode}")
        async with websockets.serve(
            self.handler,
            self.cfg.host,
            self.cfg.port,
            max_size=50 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()


def main():
    # Allow overriding via env (handy in CCI)
    CFG.port = int(os.environ.get("DA3_WS_PORT", CFG.port))
    CFG.host = os.environ.get("DA3_WS_HOST", CFG.host)
    CFG.send_mode = os.environ.get("DA3_WS_SEND_MODE", CFG.send_mode)  # jpeg_color/png16
    CFG.jpeg_quality = int(os.environ.get("DA3_WS_JPEG_QUALITY", CFG.jpeg_quality))
    CFG.target_h = int(os.environ.get("DA3_WS_H", CFG.target_h))
    CFG.target_w = int(os.environ.get("DA3_WS_W", CFG.target_w))
    CFG.amp_dtype = os.environ.get("DA3_WS_AMP", CFG.amp_dtype)  # fp16/bf16

    srv = DA3Server(CFG)
    srv.load()

    try:
        asyncio.run(srv.run())
    except KeyboardInterrupt:
        print("\n[WS] Server stopped.")


if __name__ == "__main__":
    main()
