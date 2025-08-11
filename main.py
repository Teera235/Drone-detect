#!/usr/bin/env python3
# Fast Real-Time Drone Detection (Low-Latency RTSP/Webcam)
import os, time, threading
from collections import deque
from typing import Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------ CONFIG ------------------
MODEL_PATH = "best.pt"
CAMERA_SOURCE = "rtsp://192.168.1.50:8080/h264_pcm.sdp"   # ใส่ 0 สำหรับเว็บแคม หรือ URL RTSP/HTTP จากมือถือ
IMG_SIZE = 640               # 320/416/512/640 — ยิ่งเล็กยิ่งไว
CONF_THRES = 0.5
TARGET_CLASSES = {"drone"}   # ชื่อคลาสที่สนใจ (ต้องตรงกับ model.names)
USE_GPU = torch.cuda.is_available()
HALF = True                  # ใช้ FP16 ถ้ามี GPU
DISPLAY_SCALE = 0.75         # ลดขนาดตอนแสดงผลเพื่อความลื่น
# --------------------------------------------

# ลด latency ของ ffmpeg สำหรับ RTSP (ต้อง OpenCV backend ffmpeg)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join([
    "rtsp_transport;udp",      # หรือ 'tcp' ถ้าเครือข่ายไม่นิ่ง
    "stimeout;2000000",        # 2s connect/recv timeout (ไมโครวินาที)
    "buffer_size;102400",      # ลด buffer
    "reorder_queue_size;0",
    "max_delay;0",
])

# เปิดกล้อง (รองรับทั้ง int และ URL)
def open_capture(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")

    # ลดหน่วง I/O
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass
    try: cap.set(cv2.CAP_PROP_FPS, 60)
    except: pass
    return cap

# Thread อ่านภาพ แบบ “ล่าสุดเสมอ” (ทิ้งเฟรมเก่า)
class Grabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        # รอให้มีเฟรมแรก
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 3:
            time.sleep(0.01)
        if self.frame is None:
            raise RuntimeError("No frames received from camera")

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            # ทิ้งเฟรมเก่า เก็บแต่ล่าสุด
            with self.lock:
                self.frame = f

    def get_latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1)

# โหลดโมเดลให้เร็ว
def load_model(path):
    device = "cuda:0" if USE_GPU else "cpu"
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
    model = YOLO(path)
    model.overrides["device"] = device
    model.overrides["verbose"] = False
    if USE_GPU and HALF:
        try:
            model.model.half()  # FP16
        except Exception:
            pass
    # warmup
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
    return model

def main():
    print(">>> Fast Real-Time Drone Detection starting...")
    model = load_model(MODEL_PATH)
    cap = open_capture(CAMERA_SOURCE)
    grab = Grabber(cap); grab.start()

    win = "Real-Time Detection (Low Latency)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # FPS smoother
    ticks = deque(maxlen=30)

    try:
        while True:
            frame = grab.get_latest()
            if frame is None:
                continue

            # ย่อก่อน infer เพื่อลดเวลา
            h, w = frame.shape[:2]
            scale = IMG_SIZE / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                small = frame

            # infer (ปิด verbose / เปิด imgsz)
            results = model(small, imgsz=IMG_SIZE, conf=CONF_THRES, iou=0.45, verbose=False)
            dets = []
            if results and results[0].boxes is not None:
                b = results[0].boxes
                boxes = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy()
                clss = b.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    name = model.names[clss[i]].lower()
                    if name in TARGET_CLASSES:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        # scale กลับสู่ขนาดเฟรมต้นฉบับถ้ามีการย่อ
                        if scale < 1.0:
                            inv = 1.0/scale
                            x1 = int(x1*inv); y1 = int(y1*inv); x2 = int(x2*inv); y2 = int(y2*inv)
                        dets.append((x1, y1, x2, y2, confs[i], name))

            # วาดแบบเบาที่สุด
            for (x1, y1, x2, y2, conf, name) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

            # FPS
            now = time.time(); ticks.append(now)
            fps = 0.0 if len(ticks) < 2 else (len(ticks)-1)/(ticks[-1]-ticks[0])
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

            # ลดขนาดตอนแสดงผลเพื่อความลื่น
            if DISPLAY_SCALE != 1.0:
                disp_w = int(frame.shape[1]*DISPLAY_SCALE)
                disp_h = int(frame.shape[0]*DISPLAY_SCALE)
                disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                disp = frame

            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    finally:
        grab.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
