#!/usr/bin/env python3
# Ultra-Responsive Day/Night Drone Detection (No-Blink Night) - Ready to Run
import os, time, threading
from collections import deque
from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ================== CONFIG ==================
MODEL_PATH     = "best.pt"
CAMERA_SOURCE  = "rtsp://192.168.1.35:8080/h264_pcm.sdp"   # 0 สำหรับเว็บแคม หรือ URL RTSP/HTTP
TARGET_FPS     = 24.0          # เป้าหมายความลื่นของจอ/ลูปหลัก
DAY_INIT_IMG   = 480           # ขนาดภาพเริ่มต้นสำหรับ YOLO (จะปรับขึ้น/ลงอัตโนมัติ)
DAY_MIN_IMG    = 320           # ต่ำสุดเพื่อคุมความลื่น
DAY_MAX_IMG    = 640           # สูงสุดถ้าเครื่องแรง
CONF_THRES     = 0.55
IOU_THRES      = 0.45
MAX_DET        = 40            # ลดโพสต์โปรเซส
TARGET_CLASSES = {"drone"}     # ต้องตรงกับ model.names (ตัวพิมพ์เล็ก)
USE_GPU        = torch.cuda.is_available()
USE_FP16       = True

AUTO_MODE      = True
DAY_V_MEAN     = 60.0          # mean(V) >= => Day
NIGHT_V_MEAN   = 45.0          # mean(V) <  => Night

# กลางคืน (เบาสุด)
MOG2_HISTORY    = 90
MOG2_VAR_TH     = 16
MIN_MOTION_AREA = 60
LINK_DIST       = 40
TRACK_TTL       = 0.6
HIST_POS        = 20
STAR_MAX_SPEED     = 8.0
AIRCRAFT_MIN_SPEED = 80.0
DRONE_MIN_SPEED    = 15.0
LINEARITY_HIGH     = 0.92
MAX_ROI_PER_FRAME  = 3         # จำกัดจำนวน ROI ส่งเข้า YOLO กลางคืน
INFER_BUDGET_MS    = 45.0      # ถ้าเกินงบนี้ ให้ข้ามรอบ (กันหน่วง)
DISPLAY_SCALE   = 0.85
# ============================================

# ffmpeg low-latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join([
    "rtsp_transport;udp",    # เปลี่ยนเป็น tcp ถ้า Wi-Fi ดรอป
    "stimeout;2000000",
    "buffer_size;102400",
    "reorder_queue_size;0",
    "max_delay;0",
])
try: cv2.setNumThreads(0)
except: pass

# -------- Utils --------
def mean_v(frame)->float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,2]))

def open_capture(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")
    for p,v in [(cv2.CAP_PROP_BUFFERSIZE,1),(cv2.CAP_PROP_FPS,60)]:
        try: cap.set(p,v)
        except: pass
    return cap

# Thread รับภาพแบบ “เฟรมล่าสุดเสมอ”
class Grabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.th = None
    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
        t0 = time.time()
        while self.frame is None and time.time()-t0 < 3: time.sleep(0.01)
        if self.frame is None: raise RuntimeError("No frames from camera")
    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok: time.sleep(0.003); continue
            with self.lock: self.frame = f
    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

# โหลด YOLO แบบเร็ว
def load_model(path):
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision('high')
        except: pass
    m = YOLO(path)
    m.overrides["device"] = "cuda:0" if USE_GPU else "cpu"
    m.overrides["verbose"] = False
    if USE_GPU and USE_FP16:
        try: m.model.half()
        except: pass
    return m

def class_ids_from_names(model, names:set):
    ids = [i for i,n in model.names.items() if str(n).lower() in names]
    return ids or None

# เธรดตรวจจับกลางวัน (YOLO) – ดึงเฟรมล่าสุด, ทิ้งคิว
class DayDetector:
    def __init__(self, model, class_ids, init_img, conf, iou):
        self.model = model
        self.class_ids = class_ids
        self.imgsz = init_img
        self.conf = conf
        self.iou = iou
        self.latest = None
        self.lock = threading.Lock()
        self.out_lock = threading.Lock()
        self.last = ([], 0.0)  # (dets, infer_ms)
        self.running = False
        self.th = None
    def update(self, frame):
        with self.lock: self.latest = frame
    def start(self):
        dummy = np.zeros((self.imgsz,self.imgsz,3), dtype=np.uint8)
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
            h,w = f.shape[:2]
            scale = self.imgsz / max(h,w)
            small = cv2.resize(f,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA) if scale<1.0 else f
            t0 = time.time()
            r = self.model(small, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                           classes=self.class_ids, verbose=False, max_det=MAX_DET)
            ms = (time.time()-t0)*1000.0
            if ms > INFER_BUDGET_MS:
                # งบหมด: ข้ามการอัปเดตผล เพื่อลื่น
                continue
            dets=[]
            if r and r[0].boxes is not None:
                b=r[0].boxes
                boxes=b.xyxy.cpu().numpy(); confs=b.conf.cpu().numpy(); clss=b.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    if scale<1.0:
                        inv=1/scale; x1=int(x1*inv); y1=int(y1*inv); x2=int(x2*inv); y2=int(y2*inv)
                    dets.append((x1,y1,x2,y2,float(confs[i]),int(clss[i])))
            with self.out_lock: self.last = (dets, ms)
    def get(self):
        with self.out_lock: return self.last
    def set_imgsz(self, s): self.imgsz = s
    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=1)

# กลางคืน: motion gating เบาๆ + tracking จาก centroid (ไม่พึ่งไฟ)
class SimpleTracker:
    def __init__(self):
        self.tracks: Dict[int,Dict] = {}
        self.next_id = 1
    def _d2(self,a,b):
        dx=a[0]-b[0]; dy=a[1]-b[1]; return dx*dx+dy*dy
    def update(self, pts:List[Tuple[int,int]], now:float):
        # ลบหมดอายุ
        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid]["last_t"] > TRACK_TTL:
                del self.tracks[tid]
        used=set()
        # จับคู่ของเดิม
        for tid,tr in list(self.tracks.items()):
            base=tr["pos"][-1]; best=-1; bd=1e9
            for j,p in enumerate(pts):
                if j in used: continue
                d=self._d2(base,p)
                if d<bd: bd=d; best=j
            if best!=-1 and bd<=LINK_DIST*LINK_DIST:
                p=pts[best]; used.add(best)
                tr["pos"].append(p); tr["t"].append(now); tr["last_t"]=now
                if len(tr["pos"])>HIST_POS:
                    tr["pos"].popleft(); tr["t"].popleft()
        # ของใหม่
        for j,p in enumerate(pts):
            if j in used: continue
            self.tracks[self.next_id]={
                "pos": deque([p], maxlen=HIST_POS),
                "t":   deque([now], maxlen=HIST_POS),
                "last_t": now, "label":"unknown","speed":0.0,"lin":1.0
            }
            self.next_id+=1
    def _speed(self,tr)->float:
        if len(tr["pos"])<3: return 0.0
        (x0,y0),(x1,y1) = tr["pos"][0], tr["pos"][-1]
        dt = tr["t"][-1]-tr["t"][0]
        return 0.0 if dt<=0 else float(np.hypot(x1-x0,y1-y0)/dt)
    def _linearity(self,tr)->float:
        if len(tr["pos"])<5: return 1.0
        pts=np.array(tr["pos"],dtype=np.float32)
        pts_c=pts-pts.mean(axis=0,keepdims=True)
        cov=pts_c.T@pts_c/len(pts_c)
        w,_=np.linalg.eig(cov); w=np.sort(w)[::-1]
        return float(w[0]/(w.sum()+1e-6))
    def classify(self):
        for tr in self.tracks.values():
            sp=self._speed(tr); ln=self._linearity(tr)
            tr["speed"]=sp; tr["lin"]=ln
            if sp <= STAR_MAX_SPEED: tr["label"]="star"
            elif sp >= AIRCRAFT_MIN_SPEED and ln >= LINEARITY_HIGH: tr["label"]="aircraft"
            elif sp >= DRONE_MIN_SPEED and ln < LINEARITY_HIGH: tr["label"]="drone"
            else: tr["label"]="unknown"

def night_motion_map(gray, mog2)->Tuple[np.ndarray,List[Tuple[int,int,int,int]]]:
    fg = mog2.apply(gray, learningRate=0.01)
    fg = cv2.medianBlur(fg,3)
    fg = cv2.threshold(fg, 200,255,cv2.THRESH_BINARY)[1]
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 1)
    cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if a < MIN_MOTION_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        pad=6
        rois.append((max(0,x-pad), max(0,y-pad), w+2*pad, h+2*pad))
    # จัดลำดับจากพื้นที่ใหญ่สุด แล้วตัด MAX_ROI_PER_FRAME
    rois = sorted(rois, key=lambda r:r[2]*r[3], reverse=True)[:MAX_ROI_PER_FRAME]
    return fg, rois

# ================== MAIN ==================
def main():
    print(">>> Ultra-Responsive Day/Night Drone Detection")

    model = load_model(MODEL_PATH)
    class_ids = class_ids_from_names(model, set(n.lower() for n in TARGET_CLASSES))

    cap = open_capture(CAMERA_SOURCE)
    grab = Grabber(cap); grab.start()

    day = DayDetector(model, class_ids, DAY_INIT_IMG, CONF_THRES, IOU_THRES)
    day.start()

    mog2 = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VAR_TH, detectShadows=False)
    tracker = SimpleTracker()

    win = "Drone Detection (Ultra-Responsive)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    mode = "auto" if AUTO_MODE else "day"
    fps_hist = deque(maxlen=30)
    current_img = DAY_INIT_IMG
    last_adapt = 0.0

    try:
        while True:
            t_loop = time.time()
            frame = grab.get()
            if frame is None: continue

            # กำหนดโหมด
            if mode == "auto":
                is_day = (mean_v(frame) >= DAY_V_MEAN)
            elif mode == "day":
                is_day = True
            else:
                is_day = False

            if is_day:
                # DAY: YOLO (non-blocking)
                day.update(frame)
                dets, infer_ms = day.get()

                # วาดผล
                for (x1,y1,x2,y2,conf,clsid) in dets:
                    name = str(model.names[clsid]).lower() if clsid in model.names else "obj"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{name} {conf:.0%}",(x1,max(16,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),1,cv2.LINE_AA)

                # Auto-scale imgsz ให้คง TARGET_FPS โดยไม่ทำให้ค้าง
                fps_hist.append(t_loop)
                fps = 0.0 if len(fps_hist)<2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
                if time.time()-last_adapt>0.6 and len(fps_hist)>=10:
                    last_adapt=time.time()
                    if fps < TARGET_FPS and current_img > DAY_MIN_IMG:
                        current_img = max(DAY_MIN_IMG, current_img-64); day.set_imgsz(current_img)
                    elif fps > TARGET_FPS+8 and current_img < DAY_MAX_IMG:
                        current_img = min(DAY_MAX_IMG, current_img+64); day.set_imgsz(current_img)

                status = f"DAY   | FPS:{fps:.1f} INF:{infer_ms:.1f}ms IMG:{current_img}"

            else:
                # NIGHT: เบาและลื่น — ไม่มี enhance หนัก
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg, rois = night_motion_map(gray, mog2)

                # ติดตามจาก centroid ของ motion
                cnts,_=cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pts=[]
                for c in cnts:
                    if cv2.contourArea(c) < MIN_MOTION_AREA: continue
                    x,y,w,h = cv2.boundingRect(c)
                    pts.append((x+w//2, y+h//2))
                tracker.update(pts, t_loop)
                tracker.classify()

                # YOLO เฉพาะ ROI ใหญ่สุด (จำกัด MAX_ROI_PER_FRAME)
                dets=[]
                if rois:
                    crops=[]; offs=[]
                    for (x,y,w,h) in rois:
                        crop = frame[y:y+h, x:x+w]
                        if crop.size==0: continue
                        crops.append(crop); offs.append((x,y))
                    # รันแบบมีงบเวลา: ถ้าเกิน INFER_BUDGET_MS จะข้าม
                    t0=time.time()
                    rlist = model(crops, imgsz=current_img, conf=CONF_THRES, iou=IOU_THRES,
                                  classes=class_ids, verbose=False, max_det=MAX_DET)
                    infer_ms = (time.time()-t0)*1000.0
                    if infer_ms <= INFER_BUDGET_MS:
                        for ri,(ox,oy) in zip(rlist, offs):
                            if ri.boxes is None: continue
                            b=ri.boxes
                            boxes=b.xyxy.cpu().numpy(); confs=b.conf.cpu().numpy(); clss=b.cls.cpu().numpy().astype(int)
                            for i in range(len(boxes)):
                                x1,y1,x2,y2=boxes[i].astype(int)
                                dets.append((x1+ox,y1+oy,x2+ox,y2+oy,float(confs[i]),int(clss[i])))
                else:
                    infer_ms = 0.0

                # วาด tracking + label
                for tr in tracker.tracks.values():
                    if t_loop - tr["last_t"] > TRACK_TTL: continue
                    x,y = tr["pos"][-1]
                    lab = tr["label"]; sp = tr["speed"]; ln = tr["lin"]
                    color = (0,255,0) if lab=="drone" else (255,0,0) if lab=="aircraft" else (160,160,160) if lab=="star" else (0,255,255)
                    cv2.circle(frame,(x,y),6,color,2)
                    cv2.putText(frame,f"{lab} v:{sp:.0f}px/s lin:{ln:.2f}",(x+8,y-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)

                # วาดกรอบ YOLO (ยืนยันเป้า)
                for (x1,y1,x2,y2,conf,clsid) in dets:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"drone {conf:.0%}",(x1,max(16,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),1,cv2.LINE_AA)

                # FPS display
                fps_hist.append(t_loop)
                fps = 0.0 if len(fps_hist)<2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
                status = f"NIGHT | FPS:{fps:.1f} ROI:{len(rois)} IMG:{current_img}"

            # Header
            cv2.putText(frame, status, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2,cv2.LINE_AA)

            # แสดงผลลื่น ๆ
            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1]*DISPLAY_SCALE),
                                           int(frame.shape[0]*DISPLAY_SCALE)), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            elif k == ord('d'): mode = "day"
            elif k == ord('n'): mode = "night"
            elif k == ord('a'): mode = "auto"
            elif k == ord('1'): current_img = max(DAY_MIN_IMG, current_img-64); day.set_imgsz(current_img)
            elif k == ord('2'): current_img = min(DAY_MAX_IMG, current_img+64); day.set_imgsz(current_img)

            # คุมรอบลูปหลักให้ ~TARGET_FPS (กันกิน CPU 100%)
            dt = time.time() - t_loop
            budget = max(0.0, (1.0/max(TARGET_FPS,1.0)) - dt)
            if budget > 0: time.sleep(min(budget, 0.003))

    finally:
        day.stop()
        grab.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
