#!/usr/bin/env python3
# Ultra-Low-Latency Drone Detection: RTSP/Webcam, Dual-Thread, Class-Filtered
import os, time, threading
from collections import deque
from typing import Optional, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- CONFIG (ปรับแค่ตรงนี้) ----------------
MODEL_PATH     = "best.pt"
CAMERA_SOURCE  = "rtsp://192.168.1.35:8080/h264_pcm.sdp"  # 0 สำหรับเว็บแคม หรือ URL RTSP/HTTP
INIT_IMG_SIZE  = 640     # เริ่มต้น (จะลดให้อัตโนมัติถ้า FPS ตก)
MIN_IMG_SIZE   = 320     # ย่อลงได้ต่ำสุด
TARGET_FPS     = 25.0    # เป้าหมายความลื่น
CONF_THRES     = 0.55    # ตัด false positive + ลด postprocess
IOU_THRES      = 0.45
MAX_DET        = 50
TARGET_CLASSES = {"drone"}   # ต้องตรงกับ model.names (ตัวพิมพ์เล็ก)
USE_GPU        = torch.cuda.is_available()
USE_FP16       = True
DISPLAY_SCALE  = 0.75
# ---------------------------------------------------------

# ffmpeg low-latency สำหรับ RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join([
    "rtsp_transport;udp",    # เปลี่ยนเป็น tcp ถ้า Wi-Fi สะดุด
    "stimeout;2000000",
    "buffer_size;102400",
    "reorder_queue_size;0",
    "max_delay;0",
])

# ลด contention ภายใน OpenCV
try:
    cv2.setNumThreads(0)
except:
    pass

def open_capture(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")

    # ลดหน่วง I/O (บาง backend อาจไม่รองรับ)
    for prop, val in [
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_FPS, 60),
    ]:
        try: cap.set(prop, val)
        except: pass
    return cap

def load_model(path):
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision('high')
        except: pass
    model = YOLO(path)
    model.overrides["device"] = "cuda:0" if USE_GPU else "cpu"
    model.overrides["verbose"] = False
    if USE_GPU and USE_FP16:
        try: model.model.half()
        except: pass
    return model

def class_ids_from_names(model, target_names: set) -> List[int]:
    names = model.names  # dict id->name
    ids = [i for i, n in names.items() if str(n).lower() in target_names]
    # ถ้าไม่พบชื่อใดเลย ให้ไม่กรอง (กันพัง)
    return ids if ids else None

# ----- Thread รับภาพแบบ "เฟรมล่าสุดเสมอ" -----
class Grabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.th = None

    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 3:
            time.sleep(0.01)
        if self.frame is None:
            raise RuntimeError("No frames received from camera")

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.003)
                continue
            with self.lock:
                self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

# ----- Thread อินเฟอเรนซ์: ดึงเฟรมล่าสุด, ทิ้งของเก่า -----
class InferWorker:
    def __init__(self, model, class_ids, init_imgsz, conf, iou):
        self.model = model
        self.class_ids = class_ids
        self.imgsz = init_imgsz
        self.conf = conf
        self.iou = iou
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.out_lock = threading.Lock()
        self.last_result = ([], 0.0)  # (detections, infer_ms)
        self.running = False
        self.th = None

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame

    def start(self):
        # warmup
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=self.class_ids, verbose=False)
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while self.running:
            with self.lock:
                f = None if self.latest_frame is None else self.latest_frame.copy()
                self.latest_frame = None  # ทิ้งคิวทันที
            if f is None:
                time.sleep(0.001)
                continue

            h, w = f.shape[:2]
            scale = self.imgsz / max(h, w)
            small = cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else f

            t0 = time.time()
            r = self.model(small, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                           classes=self.class_ids, verbose=False, max_det=MAX_DET)
            dt = (time.time() - t0) * 1000.0

            dets = []
            if r and r[0].boxes is not None:
                b = r[0].boxes
                boxes = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy()
                clss  = b.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    if scale < 1.0:
                        inv = 1.0/scale
                        x1 = int(x1*inv); y1 = int(y1*inv); x2 = int(x2*inv); y2 = int(y2*inv)
                    dets.append((x1, y1, x2, y2, float(confs[i]), int(clss[i])))

            with self.out_lock:
                self.last_result = (dets, dt)

    def get_result(self):
        with self.out_lock:
            return self.last_result

    def set_imgsz(self, imgsz):
        self.imgsz = imgsz  # ปรับแบบทันที frame ถัดไปจะใช้ขนาดใหม่

    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

def main():
    print(">>> Ultra-Low-Latency Drone Detection")
    model = load_model(MODEL_PATH)
    class_ids = class_ids_from_names(model, set(n.lower() for n in TARGET_CLASSES))  # กรองคลาสตั้งแต่ในโมเดล

    cap = open_capture(CAMERA_SOURCE)
    grab = Grabber(cap); grab.start()

    infer = InferWorker(model, class_ids, INIT_IMG_SIZE, CONF_THRES, IOU_THRES)
    infer.start()

    win = "Real-Time Detection (Ultra Low Latency)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # ตัวคุมความลื่นแบบง่าย: ปรับ imgsz ลงถ้า FPS ต่ำ
    fps_hist = deque(maxlen=30)
    current_imgsz = INIT_IMG_SIZE
    last_adapt_t = 0.0

    try:
        while True:
            frame = grab.get()
            if frame is None:
                continue

            infer.update_frame(frame)

            dets, infer_ms = infer.get_result()

            # วาดผล
            for (x1, y1, x2, y2, conf, clsid) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                try:
                    name = str(model.names[clsid]).lower()
                except:
                    name = "obj"
                cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(16,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

            # FPS display
            fps_hist.append(time.time())
            fps = 0.0 if len(fps_hist) < 2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
            cv2.putText(frame, f"FPS:{fps:.1f}  INF:{infer_ms:.1f}ms  IMG:{current_imgsz}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)

            # ลดขนาดการแสดงผล
            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1]*DISPLAY_SCALE),
                                           int(frame.shape[0]*DISPLAY_SCALE)),
                                   interpolation=cv2.INTER_AREA)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break
            elif k == ord('1'):
                current_imgsz = max(MIN_IMG_SIZE, current_imgsz - 64); infer.set_imgsz(current_imgsz)
            elif k == ord('2'):
                current_imgsz = min(640, current_imgsz + 64); infer.set_imgsz(current_imgsz)

            # Auto adapt imgsz ทุก ๆ 0.7s
            now = time.time()
            if now - last_adapt_t > 0.7 and len(fps_hist) >= 10:
                last_adapt_t = now
                if fps < TARGET_FPS and current_imgsz > MIN_IMG_SIZE:
                    current_imgsz = max(MIN_IMG_SIZE, current_imgsz - 64)
                    infer.set_imgsz(current_imgsz)
                elif fps > TARGET_FPS + 8 and current_imgsz < 640:
                    current_imgsz = min(640, current_imgsz + 64)
                    infer.set_imgsz(current_imgsz)

    finally:
        infer.stop()
        grab.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
