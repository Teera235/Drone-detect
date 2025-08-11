import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# ======= CONFIG =======
MODEL_PATH = "best.pt"  # ชื่อไฟล์โมเดล
CAMERA_SOURCE = "rtsp://192.168.1.35:8080/h264_pcm.sdp"  # ใส่เลขเว็บแคม เช่น 0 หรือใส่ URL RTSP/HTTP

CONF_THRESHOLD = 0.5
TARGET_CLASSES = ["drone"]  # คลาสที่สนใจ
WINDOW_NAME = "TEERATHAP INDUSTRY - Real-Time Detection"
# ======================

# โหลดโมเดล YOLO
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ตรวจว่าค่า CAMERA_SOURCE เป็น int หรือไม่
if str(CAMERA_SOURCE).isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

# เปิดกล้อง
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera source")

print("[INFO] Camera connected. Starting detection...")

fps_counter = deque(maxlen=30)
last_detections = []

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to read frame")
        continue

    # รัน YOLO ตรวจจับ
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            if model.names[clss[i]].lower() in TARGET_CLASSES:
                x1, y1, x2, y2 = boxes[i].astype(int)
                detections.append((x1, y1, x2, y2, confs[i]))

    # วาดผลลัพธ์
    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.0%}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # FPS
    fps_counter.append(time.time())
    if len(fps_counter) > 1:
        fps = (len(fps_counter)-1) / (fps_counter[-1] - fps_counter[0])
    else:
        fps = 0.0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
