# DA3-websocket-live-demo

Live Depth Anything 3 demo using **browser camera → WebSocket → GPU inference**.

Mobile / PC browsers stream camera frames to a server, which returns **real-time depth maps** with built-in frame dropping for stable low latency.


[![Live Demo Video](https://img.youtube.com/vi/lPqQSCBondA/0.jpg)](https://youtu.be/lPqQSCBondA)



<img src="https://github.com/user-attachments/assets/034285b9-f89e-402e-8eda-60e350a7aad9" width="320"/>

<sub>Mobile browser UI (live depth streaming)</sub>


---

## Prerequisite

This project is a **demo wrapper** around **Depth Anything 3**.

Please install and set up **Depth Anything 3** first:

👉 https://github.com/ByteDance-Seed/Depth-Anything-3.git

Make sure:
- DA3 runs correctly on your machine
- Model checkpoint is available locally

Set checkpoint path:

```bash
export DA3_CKPT=/path/to/da3_checkpoint.pth
````

---

## Usage

### 1. Start the WebSocket depth server (GPU machine)

```bash
python sever/server_da3_ws.py
```

This starts a WebSocket server at:

```
ws://<server_ip>:8001
```

---

### 2. Serve the demo webpage

```bash
python -m http.server 8000 --directory web
```

---

### 3. Open the demo in browser

From phone or PC:

```
http://<server_ip>:8000/index.html
```

Allow camera access → live depth will appear.

---

## Notes

* Designed for **low-latency live demos**
* Old frames are dropped automatically under load
* `out/` directory contains runtime outputs and can be ignored

---

## Optional: Video → Depth → GLB

```bash
python sever/sever_da3_glb.py
```

Then open:

```
http://<server_ip>:8000/viewer_demo.html
```

---

## Acknowledgement

This demo is built on **Depth Anything 3**:

[https://github.com/ByteDance-Seed/Depth-Anything-3.git](https://github.com/ByteDance-Seed/Depth-Anything-3.git)


