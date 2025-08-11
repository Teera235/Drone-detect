#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEERATHAP INDUSTRY - Real-Time YOLO Detector (Field-Ready Edition)
- ทนสตรีมหลุด: auto-reconnect (exponential backoff)
- หน่วงต่ำ: threaded capture + small buffer
- HUD ครบ: FPS, latency, uptime, device, drops
- กรองคลาส: ชื่อคลาสตามโมเดล
- Tracker เบาๆ: IoU-based ID tracking (ไม่ต้องติดตั้งเพิ่ม)
- Action เมื่อพบเป้าหมาย: เซฟ snapshot และ/หรือบันทึกวิดีโอ
- ใช้ได้ทั้ง RTSP/HTTP/ไฟล์/เว็บแคม
Deps: python>=3.9, opencv-python, numpy, ultralytics
"""

import os
import cv2
import time
import json
import math
import queue
import torch
import signal
import argparse
import threading
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ========================== CONFIG (ค่าเริ่มต้น) ==========================
DEFAULTS = dict(
    model_path="best.pt",
    source="rtsp://192.168.1.35:8080/h264_pcm.sdp",  # หรือ "0" สำหรับเว็บแคม
    conf=0.5,
    iou_track=0.35,               # ค่า IoU สำหรับจับคู่ track
    track_ttl=1.5,                # วินาทีที่ยอมให้ track หายก่อนลบ
    target_classes=["drone"],      # รายชื่อคลาสที่ต้องการ
    view=True,                    # แสดงหน้าต่างวิดีโอ
    record_path="",               # เว้นว่างคือไม่อัดวิดีโอ, ใส่ path เช่น "records/out.mp4"
    snapshots_dir="snapshots",    # โฟลเดอร์สำหรับบันทึกรูปเมื่อตรวจพบ
    snapshot_interval=3.0,        # อย่างน้อยกี่วินาที/ครั้ง ต่อ 1 คลาส
    line_thickness=2,             # ความหนากรอบ
    hud_font_scale=0.7,
    hud_thickness=2,
    max_fps_samples=90,           # ค่าที่ใช้คำนวณ FPS แบบวิ่ง
    max_latency_samples=120,      # เก็บ latency history สำหรับ median
    desired_width=0,              # 0 = ตามต้นทาง, >0 = บังคับความกว้าง
    desired_height=0,
    cap_backend="auto",           # "auto"|"ffmpeg"|"gstreamer"
    cap_buffer_size=1,            # ลดคิวภายในเพื่อลดดีเลย์
    reconnect_min=0.5,            # วินาที: backoff เริ่ม
    reconnect_max=5.0,            # วินาที: backoff สูงสุด
)
# ==========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Field-Ready YOLO Detector")
    p.add_argument("--model", default=DEFAULTS["model_path"])
    p.add_argument("--source", default=DEFAULTS["source"])
    p.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    p.add_argument("--classes", type=str, default=",".join(DEFAULTS["target_classes"]),
                   help="รายชื่อคลาสคั่นด้วยคอมมา เช่น 'drone,quad'")
    p.add_argument("--view", action="store_true", default=DEFAULTS["view"])
    p.add_argument("--no-view", dest="view", action="store_false")
    p.add_argument("--record", default=DEFAULTS["record_path"])
    p.add_argument("--snapshots", default=DEFAULTS["snapshots_dir"])
    p.add_argument("--snapshot-interval", type=float, default=DEFAULTS["snapshot_interval"])
    p.add_argument("--width", type=int, default=DEFAULTS["desired_width"])
    p.add_argument("--height", type=int, default=DEFAULTS["desired_height"])
    p.add_argument("--backend", choices=["auto", "ffmpeg", "gstreamer"], default=DEFAULTS["cap_backend"])
    p.add_argument("--iou-track", type=float, default=DEFAULTS["iou_track"])
    p.add_argument("--track-ttl", type=float, default=DEFAULTS["track_ttl"])
    p.add_argument("--title", default="TEERATHAP INDUSTRY - Real-Time Detection")
    return p.parse_args()


# ============================== Utilities ==============================
def is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False


def iou_xyxy(a, b):
    # a,b: (x1,y1,x2,y2)
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def draw_label(img, text, x, y, bg=(0, 0, 0), fg=(0, 255, 0), scale=0.6, th=1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (x, y - h - base - 3), (x + w + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, th, cv2.LINE_AA)


# ========================= Threaded Capture ============================
class ThreadedCapture:
    def __init__(self, source, backend="auto", width=0, height=0, buffer_size=1,
                 reconnect_min=0.5, reconnect_max=5.0):
        self.source_arg = int(source) if (isinstance(source, str) and is_int_string(source)) else source
        self.backend = backend
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.reconnect_min = reconnect_min
        self.reconnect_max = reconnect_max

        self.cap = None
        self.frame = None
        self.frame_time = 0.0
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.open()
        self.thread.start()

    def open(self):
        if self.cap:
            try:
                self.cap.release()
            except:
                pass

        if self.backend == "ffmpeg":
            cap = cv2.VideoCapture(self.source_arg, cv2.CAP_FFMPEG)
        elif self.backend == "gstreamer":
            cap = cv2.VideoCapture(self.source_arg, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.source_arg)

        # ลด buffer เพื่อลด latency (บาง backend ไม่รองรับ แต่ไม่เป็นไร)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except:
            pass

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap = cap

    def _reconnect(self, backoff):
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        time.sleep(backoff)
        self.open()

    def _loop(self):
        backoff = self.reconnect_min
        while not self.stop_flag.is_set():
            if not self.cap or not self.cap.isOpened():
                self._reconnect(backoff)
                backoff = min(self.reconnect_max, backoff * 2)
                continue

            ret, f = self.cap.read()
            now = time.time()
            if ret and f is not None:
                with self.lock:
                    self.frame = f
                    self.frame_time = now
                backoff = self.reconnect_min  # reset เมื่ออ่านสำเร็จ
            else:
                # อ่านไม่ได้ พยายามต่อใหม่
                self._reconnect(backoff)
                backoff = min(self.reconnect_max, backoff * 2)

    def get(self):
        with self.lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(), self.frame_time)

    def release(self):
        self.stop_flag.set()
        try:
            self.thread.join(timeout=1.5)
        except:
            pass
        try:
            if self.cap:
                self.cap.release()
        except:
            pass


# ============================== Tracker =================================
class IoUTracker:
    def __init__(self, iou_thr=0.35, ttl=1.5):
        self.iou_thr = iou_thr
        self.ttl = ttl
        self.tracks = {}  # id -> dict(bbox, cls, conf, last)
        self._next_id = 1

    def update(self, detections, tnow):
        """
        detections: list of dict(x1,y1,x2,y2,cls,conf)
        return: list of dict(id, x1,y1,x2,y2,cls,conf)
        """
        assigned = set()
        results = []

        # จับคู่แบบ greedy สูงสุดต่อดีเทคชัน
        for det in detections:
            best_iou, best_id = 0.0, None
            for tid, tr in self.tracks.items():
                if tr["cls"] != det["cls"]:
                    continue
                iou = iou_xyxy(tr["bbox"], (det["x1"], det["y1"], det["x2"], det["y2"]))
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= self.iou_thr:
                # อัปเดต track เดิม
                tr = self.tracks[best_id]
                tr["bbox"] = (det["x1"], det["y1"], det["x2"], det["y2"])
                tr["conf"] = float(det["conf"])
                tr["last"] = tnow
                assigned.add(best_id)
                results.append(dict(id=best_id, **det))
            else:
                # สร้าง track ใหม่
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = dict(
                    bbox=(det["x1"], det["y1"], det["x2"], det["y2"]),
                    cls=det["cls"],
                    conf=float(det["conf"]),
                    last=tnow
                )
                assigned.add(tid)
                results.append(dict(id=tid, **det))

        # ลบ track ที่หมดอายุ
        expired = [tid for tid, tr in self.tracks.items() if (tnow - tr["last"]) > self.ttl]
        for tid in expired:
            self.tracks.pop(tid, None)

        return results


# =============================== Main ===================================
def main():
    args = parse_args()
    target_classes = [s.strip().lower() for s in args.classes.split(",") if s.strip()]

    # โหลดโมเดล
    print("[INFO] Loading YOLO model:", args.model)
    model = YOLO(args.model)

    # เลือกอุปกรณ์
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda")
    print(f"[INFO] Device: {device} | FP16:{half}")

    # เตรียม I/O
    os.makedirs(args.snapshots, exist_ok=True) if args.snapshots else None

    # วิดีโอเอาต์พุต (ถ้ากำหนด)
    writer = None
    writer_fps = 25.0
    writer_size = None

    # สร้างตัวอ่านแบบเธรด
    src = args.source
    if isinstance(src, str) and is_int_string(src):
        src = int(src)
    cap = ThreadedCapture(src, backend=args.backend, width=args.width, height=args.height,
                          buffer_size=DEFAULTS["cap_buffer_size"],
                          reconnect_min=DEFAULTS["reconnect_min"], reconnect_max=DEFAULTS["reconnect_max"])

    # สถานะ
    fps_times = deque(maxlen=DEFAULTS["max_fps_samples"])
    lat_hist = deque(maxlen=DEFAULTS["max_latency_samples"])
    tracker = IoUTracker(iou_thr=args.iou_track, ttl=args.track_ttl)
    last_snapshot_at = {}  # key=(cls)->timestamp
    start_time = time.time()
    drops = 0

    # จับสัญญาณปิด
    stop_flag = {"stop": False}
    def handle_sigint(sig, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    window_name = args.title

    print("[INFO] Ready. Press 'q' or ESC to quit.")
    while not stop_flag["stop"]:
        frame, t_cap = cap.get()
        if frame is None:
            # ไม่มีเฟรม (กำลัง reconnect)
            time.sleep(0.01)
            continue

        t0 = time.time()

        # สร้าง VideoWriter เมื่อทราบขนาดเฟรม
        if writer is None and args.record:
            h, w = frame.shape[:2]
            writer_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # เขียนไฟล์ครั้งเดียว/ทับ
            os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
            writer = cv2.VideoWriter(args.record, fourcc, writer_fps, writer_size)
            print(f"[INFO] Recording -> {args.record} @ {writer_fps}fps, size={writer_size}")

        # รัน YOLO
        results = model.predict(
            source=frame, conf=args.conf, device=device, verbose=False, half=half
        )

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            b = results[0].boxes
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy().astype(int)
            names = results[0].names if hasattr(results[0], "names") else model.names

            for i in range(len(xyxy)):
                cname = str(names[clss[i]]).lower()
                if target_classes and (cname not in target_classes):
                    continue
                x1, y1, x2, y2 = xyxy[i].astype(int)
                detections.append(dict(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    cls=cname, conf=float(confs[i])
                ))

        # อัปเดต tracker
        tnow = time.time()
        tracks = tracker.update(detections, tnow)

        # วาดผล
        for det in tracks:
            x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
            conf = det["conf"]; cname = det["cls"]; tid = det["id"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), DEFAULTS["line_thickness"])
            draw_label(frame, f"{cname}#{tid} {conf:.0%}", x1, max(22, y1-8),
                       bg=(0,0,0), fg=(0,255,0), scale=0.6, th=1)

            # บันทึกรูปหลักฐานแบบ throttle ต่อคลาส
            if args.snapshots:
                key = cname
                last_t = last_snapshot_at.get(key, 0.0)
                if (tnow - last_t) >= args.snapshot_interval:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    fname = os.path.join(args.snapshots, f"{cname}_{ts}.jpg")
                    cv2.imwrite(fname, frame)
                    last_snapshot_at[key] = tnow
                    # พิมพ์ log แบบกระชับ
                    print(f"[SNAP] {fname}")

        # HUD
        t1 = time.time()
        fps_times.append(t1)
        fps = 0.0 if len(fps_times) < 2 else (len(fps_times)-1) / (fps_times[-1] - fps_times[0])
        latency_ms = (t1 - t0) * 1000.0
        lat_hist.append(latency_ms)
        med_lat = np.median(lat_hist) if lat_hist else 0.0
        uptime = t1 - start_time

        hud_lines = [
            f"FPS: {fps:4.1f} | Latency: {latency_ms:5.1f} ms (med {med_lat:4.1f})",
            f"Device: {device} | Tracks: {len(tracker.tracks)}",
            f"Uptime: {uptime:6.1f}s | Drops: {drops}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        y = 28
        for line in hud_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        DEFAULTS["hud_font_scale"], (0,255,0), DEFAULTS["hud_thickness"], cv2.LINE_AA)
            y += 26

        # แสดงผล/บันทึก
        if args.view:
            cv2.imshow(args.title, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break

        if writer is not None:
            try:
                writer.write(frame)
            except:
                drops += 1

    # ปิดโปรแกรม
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
