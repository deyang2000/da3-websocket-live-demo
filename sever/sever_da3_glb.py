#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import uuid
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from aiohttp import web

from depth_anything_3.api import DepthAnything3

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    ckpt: str = "/mnt/afs/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"  # HF name or local dir
    device: str = "cuda"
    max_frames: int = 500
    out_dir: str = "./out"

CFG = Config()

INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DA3 录制 → GLB 查看</title>
  
    <script type="importmap">
    {
    "imports": {
        "three": "/static/three/three.module.js",
        "three/examples/jsm/loaders/GLTFLoader.js": "/static/three/GLTFLoader.js",
        "three/examples/jsm/controls/OrbitControls.js": "/static/three/OrbitControls.js",
        "three/examples/jsm/utils/BufferGeometryUtils.js": "/static/three/BufferGeometryUtils.js"
    }
    }
    </script>

<style>
  /* 1. 变量定义 */
  :root {
    --bg-main: #0b1020;
    --bg-panel: rgba(255, 255, 255, 0.04);
    --bg-card: rgba(255, 255, 255, 0.06);
    --border-soft: rgba(255, 255, 255, 0.08);
    --accent: #4da3ff;
    --accent-2: #8b5cf6;
    --accent-ok: #22c55e;
    --text-main: #e5e7eb;
    --text-dim: #9ca3af;
  }

  /* 2. 基础布局与背景 */
  body {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background:
      radial-gradient(1200px 600px at 70% -10%, #1b255a55, transparent),
      linear-gradient(180deg, #070b18, #0b1020);
    color: var(--text-main);
    overflow: hidden; /* 防止页面整体滚动 */
  }

  .wrap {
    display: grid;
    grid-template-columns: 360px 1fr;
    height: 100vh;
  }

  /* 3. 左侧面板（带毛玻璃效果） */
  .panel {
    padding: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border-right: 1px solid var(--border-soft);
    backdrop-filter: blur(12px);
    overflow-y: auto;
  }

  /* 4. 卡片样式 */
  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  }

  /* 5. 交互组件 */
  button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border: none;
    color: white;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: transform .15s ease, box-shadow .15s ease, opacity .2s;
  }

  button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(77,163,255,0.35);
  }

  button:active:not(:disabled) {
    transform: translateY(1px);
  }

  button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    background: var(--text-dim);
  }

  button.secondary {
    background: rgba(255,255,255,0.1);
  }

  video {
    width: 100%;
    border-radius: 12px;
    background: #000;
    margin-top: 12px;
    border: 1px solid var(--border-soft);
  }

  /* 6. 状态与预览区 */
  #status {
    font-family: ui-monospace, 'Cascadia Code', Menlo, monospace;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-dim);
    white-space: pre-wrap;
    word-break: break-all;
  }

  #view {
    position: relative;
    background: #000; /* 3D区底色 */
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  /* 进度条样式 */
  .progress-wrap {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    margin: 12px 0;
    overflow: hidden;
    display: none;
  }
  #progressBar {
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-ok));
    transition: width 0.3s ease;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div class="card">
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btnStart">开始录制(10s)</button>
        <button id="btnStop" class="secondary" disabled>停止</button>
        <button id="btnUpload" disabled>上传并生成GLB</button>
      </div>
      <div style="margin-top:8px;font-size:12px;opacity:.85">
        右侧：拖拽旋转，滚轮缩放（GLB加载）。
      </div>
      <div style="margin-top:8px">
        <video id="preview" autoplay playsinline muted></video>
      </div>
    </div>
    <div class="card">
      <div id="status">Ready.</div>
    </div>
  </div>
  <div id="view"></div>
</div>

<script type="module">
  import * as THREE from "three"; 
  import { OrbitControls } from "/static/three/OrbitControls.js";
  import { GLTFLoader } from "/static/three/GLTFLoader.js";

  const statusEl = document.getElementById('status');
  const log = (s) => statusEl.textContent = s;

  // recording
  const preview = document.getElementById('preview');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnUpload = document.getElementById('btnUpload');

  let stream=null, rec=null, chunks=[], recordedBlob=null;
  let useRearCamera = true;



  async function ensureStream() {
    if (stream) return stream;

    const constraints = {
      video: {
        facingMode: useRearCamera ? { ideal: "environment" } : { ideal: "user" }
      },
      audio: false
    };

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    preview.srcObject = stream;
    return stream;
  }



  btnStart.onclick = async () => {
    try{
      await ensureStream();
      chunks=[]; recordedBlob=null;

      const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
      rec = new MediaRecorder(stream, { mimeType: mime });

      rec.ondataavailable = (e)=>{ if(e.data && e.data.size>0) chunks.push(e.data); };
      rec.onstop = ()=>{
        recordedBlob = new Blob(chunks, { type: rec.mimeType });
        btnUpload.disabled = false;
        log(`Recorded: ${(recordedBlob.size/1024/1024).toFixed(2)} MB`);
      };

      rec.start();
      btnStart.disabled=true; btnStop.disabled=false; btnUpload.disabled=true;
      log("Recording 10 seconds...");
      setTimeout(()=>{ if(rec && rec.state==='recording') rec.stop(); }, 10000);
    }catch(e){ log("Recording error: "+e); }
  };

  btnStop.onclick = ()=> {
    if(rec && rec.state==='recording') rec.stop();
    btnStop.disabled=true; btnStart.disabled=false;
  };

  btnUpload.onclick = async ()=> {
    if(!recordedBlob) return;
    btnUpload.disabled=true;
    log("Uploading video and exporting GLB...");

    const form = new FormData();
    form.append('video', recordedBlob, 'clip.webm');

    const resp = await fetch('/api/upload', { method:'POST', body: form });
    if(!resp.ok){ log("Upload failed: "+resp.status); return; }
    const js = await resp.json();
    log("GLB ready: "+js.glb_url);

    await loadGLB(js.glb_url);
    btnUpload.disabled=false;
  };

  // three viewer
  const view = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ antialias:true });
  view.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d12);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, -1.2, 0.9);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;

  scene.add(new THREE.AxesHelper(0.2));
  scene.add(new THREE.GridHelper(4, 40));
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));

  let current = null;
 

  const loader = new GLTFLoader();

  function resize(){
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w/h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  async function loadGLB(url){
    return new Promise((resolve, reject)=>{
      loader.load(url, (gltf)=>{
        if(current){
          scene.remove(current);
          current = null;
        }
        current = gltf.scene;
        scene.add(current);

        // try to frame it
        const box = new THREE.Box3().setFromObject(current);
        const size = new THREE.Vector3(); box.getSize(size);
        const center = new THREE.Vector3(); box.getCenter(center);
        current.position.sub(center);
        orbit.target.set(0,0,0);

        const r = Math.max(size.x, size.y, size.z);
        camera.position.set(0, -Math.max(1.2, r*1.5), Math.max(0.8, r*1.2));

        resolve();
      }, undefined, reject);
    });
  }

  function animate(){
    requestAnimationFrame(animate);
    orbit.update();
    renderer.render(scene, camera);
  }
  animate();
</script>
</body>
</html>
"""
import subprocess
import tempfile
from pathlib import Path

def sample_frames_ffmpeg(video_path: str, max_frames: int):
    """
    使用 ffmpeg 均匀抽取最多 max_frames 帧，返回 RGB numpy list
    """
    video_path = Path(video_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="da3_frames_"))

    # 抽帧（ffmpeg 内部会按时间顺序）
    out_pattern = str(tmp_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={max_frames}/20",  # 假设最长 20s，可按需改
        "-q:v", "2",
        out_pattern
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frames = []
    for p in sorted(tmp_dir.glob("frame_*.jpg")):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    return frames

def sample_frames(video_path: str, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        idxs = list(range(max_frames))
    else:
        idxs = np.linspace(0, max(0, total-1), num=max_frames, dtype=np.int32).tolist()

    frames_rgb = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
    cap.release()
    return frames_rgb

class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print("[DA3] loading:", cfg.ckpt, "device:", device)
        self.model = DepthAnything3.from_pretrained(cfg.ckpt).to(device=device)

    async def index(self, request):
        return web.Response(text=INDEX_HTML, content_type="text/html", charset="utf-8")


    async def upload(self, request):
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "video":
            return web.json_response({"error":"missing video"}, status=400)

        job_id = uuid.uuid4().hex[:10]
        job_dir = self.out_dir / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        video_path = job_dir / "clip.webm"
        with open(video_path, "wb") as f:
            while True:
                chunk = await part.read_chunk(size=1<<20)
                if not chunk:
                    break
                f.write(chunk)

        t0 = time.time()
        frames = sample_frames_ffmpeg(str(video_path), self.cfg.max_frames)
        print(f"[DEBUG] extracted {len(frames)} frames from video")

        if len(frames) == 0:
            return web.json_response({"error":"no frames decoded"}, status=400)

        # Official inference + export
        pred = self.model.inference(
            frames,
            export_dir=str(job_dir),
            export_format="glb"
        )

        # find glb in job_dir
        glbs = list(job_dir.glob("*.glb"))
        if not glbs:
            # some implementations put results under export_dir/subdir, broaden search
            glbs = list(job_dir.rglob("*.glb"))
        if not glbs:
            return web.json_response({"error":"glb not generated"}, status=500)

        glb_path = glbs[0]
        dt = time.time() - t0
        print(f"[GLB] job={job_id} frames={len(frames)} glb={glb_path.name} time={dt:.2f}s")

        return web.json_response({
            "glb_url": f"/out/job_{job_id}/{glb_path.name}",
            "frames_used": len(frames),
            "time_sec": dt,
            "depth_shape": list(pred.depth.shape) if hasattr(pred, "depth") else None
        })

    async def serve_out(self, request):
        rel = request.match_info["path"]
        path = (self.out_dir / rel).resolve()
        if not str(path).startswith(str(self.out_dir.resolve())):
            return web.Response(status=403, text="forbidden")
        if not path.exists():
            return web.Response(status=404, text="not found")
        return web.FileResponse(path)

    def make_app(self):
        app = web.Application(client_max_size=200*1024*1024)
        
        # 确保静态文件路径是绝对路径，防止找不到
        static_path = Path(__file__).parent / "static"
        static_path.mkdir(exist_ok=True) 

        app.add_routes([
            web.get("/", self.index),
            web.post("/api/upload", self.upload),
            # 1. 注册静态资源路由，将 /static 映射到本地的 static 文件夹
            web.static("/static", str(static_path)), 
            web.get("/out/{path:.*}", self.serve_out),
        ])
        return app

def main():
    CFG.port = int(os.environ.get("DA3_GLB_PORT", CFG.port))
    CFG.host = os.environ.get("DA3_GLB_HOST", CFG.host)
    CFG.ckpt = os.environ.get("DA3_GLB_CKPT", CFG.ckpt)

    app = App(CFG)
    web.run_app(app.make_app(), host=CFG.host, port=CFG.port)

if __name__ == "__main__":
    main()
