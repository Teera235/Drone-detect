#!/usr/bin/env python3
# Day/Night Drone Detection (Real-Time, Low-Latency, Ready-to-Use)
import os, time, threading
from collections import deque
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------- CONFIG (แก้แค่ตรงนี้) -------------
MODEL_PATH     = "best.pt"
CAMERA_SOURCE  = "rtsp://192.168.1.35:8080/h264_pcm.sdp"   # ใส่ 0 สำหรับเว็บแคม หรือ URL RTSP/HTTP
INIT_IMG_SIZE  = 640
MIN_IMG_SIZE   = 320
TARGET_FPS     = 25.0
CONF_THRES     = 0.55
IOU_THRES      = 0.45
MAX_DET        = 50
TARGET_CLASSES = {"drone"}   # ต้องตรงกับ model.names
USE_GPU        = torch.cuda.is_available()
USE_FP16       = True
DISPLAY_SCALE  = 0.8

# การตัดสินใจ Day/Night
AUTO_MODE      = True
DAY_BRIGHT_V   = 60          # ค่า V (0-255) เฉลี่ยเกินนี้ → Day
NIGHT_BRIGHT_V = 45          # ต่ำกว่านี้ → Night

# Night heuristics
AREA_RANGE     = (1, 300)    # พิกเซลของจุดแสง
LINK_DIST      = 30          # px สำหรับจับคู่จุดเดิม
TRACK_TTL      = 0.7         # วินาที, ไม่มีอัปเดต = ลบทิ้ง
HIST_LEN       = 48          # จำนวน sample สร้างสเปกตรัม
FREQ_RANGE     = (0.6, 8.0)  # Hz ที่ถือว่าเป็น "กะพริบ"
AIR_HZ_RANGE   = (0.6, 2.0)  # โดยทั่วไปเครื่องบิน ~1Hz
DRONE_MIN_HZ   = 2.0         # โดรนมักกะพริบเร็วกว่า
SPEED_THRESH   = 40          # px/s แยก star vs moving
# ---------------------------------------------------

# ffmpeg low-latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join([
    "rtsp_transport;udp",  # เปลี่ยนเป็น tcp ถ้า Wi-Fi ดรอป
    "stimeout;2000000",
    "buffer_size;102400",
    "reorder_queue_size;0",
    "max_delay;0",
])

try: cv2.setNumThreads(0)
except: pass

# ---------- I/O ----------
def open_capture(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")
    for prop, val in [(cv2.CAP_PROP_BUFFERSIZE, 1), (cv2.CAP_PROP_FPS, 60)]:
        try: cap.set(prop, val)
        except: pass
    return cap

class Grabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.th = None
    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
        t0 = time.time()
        while self.frame is None and time.time()-t0 < 3:
            time.sleep(0.01)
        if self.frame is None: raise RuntimeError("No frames from camera")
    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.003); continue
            with self.lock: self.frame = f
    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

# ---------- Day pipeline (YOLO) ----------
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
    names = model.names
    ids = [i for i, n in names.items() if str(n).lower() in target_names]
    return ids if ids else None

class InferWorker:
    def __init__(self, model, class_ids, init_imgsz, conf, iou):
        self.model = model
        self.class_ids = class_ids
        self.imgsz = init_imgsz
        self.conf = conf
        self.iou = iou
        self.latest: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.out_lock = threading.Lock()
        self.last = ([], 0.0)  # (dets, ms)
        self.running = False
        self.th = None
    def update(self, frame):
        with self.lock: self.latest = frame
    def start(self):
        # warmup
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                       classes=self.class_ids, verbose=False, max_det=MAX_DET)
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
    def _loop(self):
        while self.running:
            with self.lock:
                f = None if self.latest is None else self.latest.copy()
                self.latest = None
            if f is None:
                time.sleep(0.001); continue
            h, w = f.shape[:2]
            scale = self.imgsz / max(h, w)
            small = cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else f
            t0 = time.time()
            r = self.model(small, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                           classes=self.class_ids, verbose=False, max_det=MAX_DET)
            ms = (time.time()-t0)*1000.0
            dets = []
            if r and r[0].boxes is not None:
                b = r[0].boxes
                boxes = b.xyxy.cpu().numpy(); confs = b.conf.cpu().numpy(); clss = b.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    if scale < 1.0:
                        inv=1/scale; x1=int(x1*inv);y1=int(y1*inv);x2=int(x2*inv);y2=int(y2*inv)
                    dets.append((x1,y1,x2,y2,float(confs[i]),int(clss[i])))
            with self.out_lock: self.last = (dets, ms)
    def get(self): 
        with self.out_lock: return self.last
    def set_imgsz(self, s): self.imgsz = s
    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

# ---------- Night pipeline (light blinking + tracking) ----------
class Track:
    def __init__(self, tid:int, xy:Tuple[int,int], t:float, bright:float):
        self.id = tid
        self.pos = deque(maxlen=HIST_LEN)      # (x,y)
        self.tms = deque(maxlen=HIST_LEN)      # timestamps
        self.bri = deque(maxlen=HIST_LEN)      # brightness
        self.last = t
        self.pos.append(xy); self.tms.append(t); self.bri.append(bright)
        self.label = "unknown"
        self.freq = 0.0
        self.speed = 0.0
    def update(self, xy, t, bright):
        self.pos.append(xy); self.tms.append(t); self.bri.append(bright); self.last = t
    def alive(self, now): return (now - self.last) <= TRACK_TTL

class NightDetector:
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _threshold_bright(self, gray):
        # ใช้เปอร์เซ็นไทล์เพื่ออะแดปต์ตามสภาพ
        p99 = float(np.percentile(gray, 99.0))
        thr = max(200.0, p99 - 5.0)
        _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        # เคลียร์ noise เล็กๆ
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        return bw

    def _extract_points(self, frame, bw):
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H,W = frame.shape[:2]
        pts = []
        for c in cnts:
            a = cv2.contourArea(c)
            if AREA_RANGE[0] <= a <= AREA_RANGE[1]:
                x,y,w,h = cv2.boundingRect(c)
                cx, cy = x + w//2, y + h//2
                roi = frame[max(0,y-2):min(H,y+h+2), max(0,x-2):min(W,x+w+2)]
                bright = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:,:,2])) if roi.size>0 else 0.0
                pts.append(((cx,cy),(x,y,x+w,y+h), bright))
        return pts

    def _link_tracks(self, pts, tnow):
        # ลบ track หมดอายุ
        self.tracks = {tid:tr for tid,tr in self.tracks.items() if tr.alive(tnow)}

        used = set()
        # จับคู่จากใกล้สุด
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = -1, 1e9
            px,py = tr.pos[-1]
            for j,(xy,bb,bri) in enumerate(pts):
                if j in used: continue
                d = (xy[0]-px)**2 + (xy[1]-py)**2
                if d < best_d:
                    best_d, best_j = d, j
            if best_j>=0 and best_d <= LINK_DIST**2:
                xy, bb, bri = pts[best_j]
                tr.update(xy, tnow, bri)
                used.add(best_j)

        # สร้าง track ใหม่สำหรับจุดที่เหลือ
        for j,(xy,bb,bri) in enumerate(pts):
            if j in used: continue
            tr = Track(self.next_id, xy, tnow, bri)
            self.tracks[self.next_id] = tr
            self.next_id += 1

    def _dominant_freq(self, tr:Track) -> Tuple[float,float]:
        # ประมาณ sample rate จากเวลา
        if len(tr.tms) < 8: return 0.0, 0.0
        dt = (tr.tms[-1]-tr.tms[0])/(len(tr.tms)-1) if len(tr.tms)>1 else 0.0
        if dt <= 0: return 0.0, 0.0
        fs = 1.0/dt
        x = np.array(tr.bri, dtype=np.float32)
        x = x - x.mean()
        if np.allclose(x, 0): return 0.0, 0.0
        # FFT
        n = len(x)
        # สร้างหน้าต่างลด leakage
        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * win)
        freqs = np.fft.rfftfreq(n, d=dt)
        mag = np.abs(X)
        # สนใจช่วงกะพริบเท่านั้น
        mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
        if not np.any(mask): return 0.0, 0.0
        idx = np.argmax(mag[mask])
        f0 = float(freqs[mask][idx]); a0 = float(mag[mask][idx])
        return f0, a0

    def _speed_pxps(self, tr:Track) -> float:
        if len(tr.pos) < 3: return 0.0
        dx = tr.pos[-1][0] - tr.pos[0][0]
        dy = tr.pos[-1][1] - tr.pos[0][1]
        dt = tr.tms[-1] - tr.tms[0]
        if dt <= 0: return 0.0
        return float(np.hypot(dx,dy) / dt)

    def _classify(self, tr:Track):
        f0, a0 = self._dominant_freq(tr)
        spd = self._speed_pxps(tr)
        tr.freq, tr.speed = f0, spd

        # กฎง่าย ๆ:
        # - STAR: ไม่กะพริบเด่น (<0.6Hz หรือแอมป์ต่ำ) และความเร็วต่ำ
        # - AIRCRAFT: f ~ 0.6–2Hz และวิ่งเร็ว/เส้นทางยาว
        # - DRONE: f >= 2Hz หรือเคลื่อนที่ช้า/แกว่ง ๆ ใกล้กล้อง
        amp_ok = a0 > 1.0  # กรองสัญญาณกะพริบจริง ๆ
        if (f0 < AIR_HZ_RANGE[0] or not amp_ok) and spd < SPEED_THRESH:
            tr.label = "star"
        elif AIR_HZ_RANGE[0] <= f0 <= AIR_HZ_RANGE[1] and spd >= SPEED_THRESH*0.8:
            tr.label = "aircraft"
        elif f0 >= DRONE_MIN_HZ or (spd < SPEED_THRESH and amp_ok):
            tr.label = "drone"
        else:
            tr.label = "unknown"

    def step(self, frame: np.ndarray, tnow: float):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        bw = self._threshold_bright(gray)
        pts = self._extract_points(frame, bw)
        self._link_tracks(pts, tnow)
        # จัดประเภท
        for tr in self.tracks.values():
            if len(tr.bri) >= 12:
                self._classify(tr)
        # ส่งออก bbox ประมาณจากตำแหน่ง (สำหรับวาด)
        out = []
        for tr in self.tracks.values():
            if not tr.alive(tnow): continue
            x,y = tr.pos[-1]
            out.append({
                "id": tr.id, "xy": (x,y),
                "label": tr.label, "freq": tr.freq, "speed": tr.speed
            })
        return out, bw

# ---------- Utils ----------
def mean_v(frame) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,2]))

# ---------- Main ----------
def main():
    print(">>> Day/Night Drone Detection (Auto)")
    model = load_model(MODEL_PATH)
    class_ids = class_ids_from_names(model, set(n.lower() for n in TARGET_CLASSES))

    cap = open_capture(CAMERA_SOURCE)
    grab = Grabber(cap); grab.start()

    infer = InferWorker(model, class_ids, INIT_IMG_SIZE, CONF_THRES, IOU_THRES)
    infer.start()

    night = NightDetector()
    win = "Drone Detection (Day/Night Auto)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    mode = "auto" if AUTO_MODE else "day"
    fps_hist = deque(maxlen=30)
    current_imgsz = INIT_IMG_SIZE
    last_adapt_t = 0.0

    try:
        while True:
            frame = grab.get()
            if frame is None:
                continue
            now = time.time()

            # ตัดสิน Day/Night
            if mode == "auto":
                v = mean_v(frame)
                is_day = v >= DAY_BRIGHT_V
            elif mode == "day":
                is_day = True
            else:
                is_day = False

            if is_day:
                # DAY: YOLO
                infer.update(frame)
                dets, infer_ms = infer.get()

                # วาดผล
                for (x1,y1,x2,y2,conf,clsid) in dets:
                    name = str(model.names[clsid]).lower() if clsid in model.names else "obj"
                    color = (0,255,0) if name=="drone" else (255,255,255)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(16,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                # ปรับ imgsz เพื่อความลื่น
                fps_hist.append(now)
                fps = 0.0 if len(fps_hist)<2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
                if time.time()-last_adapt_t>0.7 and len(fps_hist)>=10:
                    last_adapt_t = time.time()
                    if fps < TARGET_FPS and current_imgsz > MIN_IMG_SIZE:
                        current_imgsz = max(MIN_IMG_SIZE, current_imgsz-64); infer.set_imgsz(current_imgsz)
                    elif fps > TARGET_FPS+8 and current_imgsz < 640:
                        current_imgsz = min(640, current_imgsz+64); infer.set_imgsz(current_imgsz)

                mode_txt = f"DAY  | FPS:{fps:.1f}  INF:{infer_ms:.1f}ms  IMG:{current_imgsz}"

            else:
                # NIGHT: จุดแสงกะพริบ + ติดตาม + จัดประเภท
                tracks, bw = night.step(frame, now)
                for t in tracks:
                    x,y = t["xy"]; label=t["label"]; freq=t["freq"]; spd=t["speed"]
                    if label=="drone": color=(0,255,0)
                    elif label=="aircraft": color=(255,0,0)
                    elif label=="star": color=(150,150,150)
                    else: color=(0,255,255)
                    cv2.circle(frame, (x,y), 6, color, 2)
                    cv2.putText(frame, f"{label} f:{freq:.1f}Hz v:{spd:.0f}px/s",
                                (x+8,y-8), cv2.FONT_SANS_SERIF, 0.45, color, 1, cv2.LINE_AA)

                fps_hist.append(now)
                fps = 0.0 if len(fps_hist)<2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
                mode_txt = f"NIGHT | FPS:{fps:.1f}  Tracks:{len(tracks)}"

            # แสดงสถานะ
            cv2.putText(frame, mode_txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)

            # ลดขนาดการแสดงผล
            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1]*DISPLAY_SCALE),
                                           int(frame.shape[0]*DISPLAY_SCALE)),
                                   interpolation=cv2.INTER_AREA)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            elif k == ord('d'): mode = "day"
            elif k == ord('n'): mode = "night"
            elif k == ord('a'): mode = "auto"

    finally:
        infer.stop()
        grab.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
