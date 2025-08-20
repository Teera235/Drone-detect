#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline YOLO Inference for Drone Detection
- เลือกไฟล์วิดีโอ (GUI) หรือส่ง --input
- ประมวลผลทั้งคลิปแบบออฟไลน์ (ไม่สตรีม)
- เขียนวีดีโอผลลัพธ์ MP4 พร้อมกล่อง/ฉลาก
- เสร็จแล้วเปิดไฟล์อัตโนมัติ
- ตัวเลือกคัดลอกเสียงด้วย ffmpeg (--mux-audio)

ต้องมี: ultralytics, opencv-python, torch, tqdm
"""

import os, sys, time, argparse, shutil, subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    print("ไม่พบ ultralytics: ติดตั้งด้วย `pip install ultralytics`")
    raise

# ---------- GUI file picker (tkinter) ----------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอ",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv"), ("All files", "*.*")]
        )
        return path if path else None
    except Exception:
        return None  # ถ้า GUI ใช้ไม่ได้ ให้ผู้ใช้ส่ง --input แทน

# ---------- open file with OS default ----------
def open_with_os(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass

# ---------- optional: mux audio using ffmpeg ----------
def mux_audio(src_video: Path, dst_video_noaudio: Path, dst_with_audio: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y",
        "-i", str(dst_video_noaudio),
        "-i", str(src_video),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(dst_with_audio)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

# ---------- draw boxes ----------
def draw_detections(frame: np.ndarray, result, names, show_score=True):
    if result.boxes is None: 
        return frame
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    h, w = frame.shape[:2]
    thick = max(2, int(0.002 * (h + w)))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        c = cls[i]
        score = conf[i]
        label = names.get(c, str(c))
        # กล่อง + เงา label สไตล์เรียบ
        color = (40, 220, 40) if "drone" in label.lower() else (50, 180, 255)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, thick)

        text = f"{label}" + (f" {score:.2f}" if show_score else "")
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        cv2.rectangle(frame, (p1[0], p1[1]-th-8), (p1[0]+tw+6, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0]+3, p1[1]-5), font, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Offline YOLO Drone Detection (render full clip then open)")
    parser.add_argument("--input", type=str, default=None, help="path วิดีโอ (ถ้าไม่ใส่จะมีหน้าต่างให้เลือก)")
    parser.add_argument("--model", type=str, default="best.pt", help="path โมเดล YOLO (ค่าเริ่มต้น: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    parser.add_argument("--iou",  type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--device", type=str, default=None, help="cuda หรือ cpu (default: auto)")
    parser.add_argument("--open", action="store_true", help="เปิดไฟล์ผลลัพธ์อัตโนมัติเมื่อเสร็จ")
    parser.add_argument("--mux-audio", action="store_true", help="พยายามคัดลอกเสียงเดิมด้วย ffmpeg")
    parser.add_argument("--half", action="store_true", help="ใช้ half-precision ถ้า GPU รองรับ")
    parser.add_argument("--classes", type=str, default=None, help="คัดกรองคลาส เช่น '0,1' หรือ 'drone,person'")
    args = parser.parse_args()

    # เลือกไฟล์ถ้าไม่ส่ง --input
    in_path = args.input or pick_file()
    if not in_path:
        print("ไม่ได้เลือกไฟล์ และไม่ได้ส่ง --input ยุติการทำงาน")
        sys.exit(1)

    in_path = Path(in_path)
    if not in_path.exists():
        print(f"ไม่พบไฟล์: {in_path}")
        sys.exit(1)

    # โหลดโมเดล
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ไม่พบโมเดล: {model_path}")
        sys.exit(1)
    model = YOLO(str(model_path))

    # เลือกอุปกรณ์
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # แปลง classes ที่กรอง
    class_filter = None
    if args.classes:
        raw = [x.strip() for x in args.classes.split(",") if x.strip()]
        # แปลง string เป็น index ถ้าเป็นตัวเลข
        names = model.names
        name_to_idx = {v: k for k, v in names.items()}
        class_filter = []
        for r in raw:
            if r.isdigit():
                class_filter.append(int(r))
            else:
                # ชื่อคลาส
                if r in name_to_idx:
                    class_filter.append(name_to_idx[r])
                else:
                    # เผื่อกรณี r เป็น "drone" แต่ชื่อจริงไม่ตรง
                    # จะลองจับแบบ lower-contains ตัวแรกที่เจอ
                    m = [k for k, v in names.items() if r.lower() in v.lower()]
                    class_filter.extend(m)

    # เปิดวิดีโอ
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print("เปิดวิดีโอไม่สำเร็จ")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # เตรียมที่เขียนไฟล์
    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = out_dir / f"processed_{stamp}"
    out_mp4_noaudio = out_base.with_suffix(".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_noaudio), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("เปิด VideoWriter ไม่ได้ ลองเปลี่ยน fourcc หรือนามสกุลไฟล์")
        sys.exit(1)

    names = model.names

    # ค่าคงที่ inference
    predict_kwargs = dict(conf=args.conf, iou=args.iou, device=device, verbose=False, imgsz=max(640, max(w, h)))
    # half precision ถ้าได้
    if args.half:
        predict_kwargs["half"] = True

    pbar = tqdm(total=nF if nF else 0, unit="f", disable=(nF is None))
    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # inference
            res = model(frame, **predict_kwargs)[0]

            # กรองคลาสถ้าระบุ
            if class_filter is not None and res.boxes is not None and len(res.boxes) > 0:
                keep = []
                for i, c in enumerate(res.boxes.cls.cpu().numpy().astype(int)):
                    if c in class_filter:
                        keep.append(i)
                if len(keep) == 0:
                    # เขียนเฟรมเดิม
                    writer.write(frame)
                    frame_idx += 1
                    if nF: pbar.update(1)
                    continue
                # เลือกเฉพาะ box ที่ต้องการ
                res.boxes = res.boxes[keep]

            # วาดผล
            out_frame = draw_detections(frame, res, names, show_score=True)
            writer.write(out_frame)

            frame_idx += 1
            if nF: pbar.update(1)
    finally:
        cap.release()
        writer.release()
        pbar.close()

    # รวมเสียง (ถ้าขอและมี ffmpeg)
    final_out = out_mp4_noaudio
    if args.mux-audio:
        out_with_audio = out_base.with_name(out_base.name + "_audio").with_suffix(".mp4")
        if mux_audio(in_path, out_mp4_noaudio, out_with_audio):
            final_out = out_with_audio

    dt = time.time() - t0
    print(f"\nเสร็จแล้ว -> {final_out}  ({frame_idx} เฟรม, {dt:.1f}s, ~{(frame_idx/max(dt,1e-9)):.1f} fps เขียนออก)")

    if args.open:
        open_with_os(final_out)

if __name__ == "__main__":
    main()
