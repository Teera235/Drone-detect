import cv2
import time
import numpy as np
from ultralytics import YOLO

# โหลดโมเดลและบังคับใช้ CPU
model = YOLO('best.pt')

# เปิดวีดีโอ
video_path = "videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# ลด buffer เพื่อลด delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ตัวแปรควบคุม
frame_count = 0
prev_time = time.time()
process_every_n_frames = 3
last_detection = None

# ตัวแปรสำหรับตรวจจับแสงสว่าง
brightness_threshold = 200  # ความสว่างขั้นต่ำ (0-255)
min_area = 50              # ขนาดพื้นที่ขั้นต่ำของแสงสว่าง
max_area = 5000            # ขนาดพื้นที่สูงสุด

def detect_bright_spots(frame):
    """ฟังก์ชันสำหรับตรวจจับจุดแสงสว่าง"""
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Gaussian blur เพื่อลด noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # สร้าง threshold mask สำหรับพื้นที่สว่าง
    _, thresh = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # หา contours ของพื้นที่สว่าง
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bright_spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # คำนวณ center และ bounding box
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # สร้าง bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bright_spots.append({
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'area': area
                })
    
    return bright_spots, thresh

def draw_bright_detections(frame, bright_spots):
    """วาด bounding box สำหรับแสงสว่างที่ตรวจจับได้"""
    result_frame = frame.copy()
    
    for spot in bright_spots:
        x, y, w, h = spot['bbox']
        cx, cy = spot['center']
        area = spot['area']
        
        # วาด bounding box (สีส้มเพื่อแยกจาก YOLO)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        
        # วาด center point
        cv2.circle(result_frame, (cx, cy), 5, (0, 165, 255), -1)
        
        # เพิ่มข้อความ
        label = f"Bright Spot (Area: {int(area)})"
        cv2.putText(result_frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    return result_frame

print("เริ่มการตรวจจับ (YOLO + Bright Light Detection)...")
print("กด 'q' เพื่อออก | กด '+/-' ปรับ frame skip")
print("กด 'b' เพื่อเพิ่มความสว่าง threshold | กด 'v' เพื่อลดความสว่าง threshold")
print("กด 'd' เพื่อดู threshold mask")

show_threshold = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    original_h, original_w = frame.shape[:2]
    yolo_detected = False
    
    # ประมวลผล YOLO เฉพาะเฟรมที่กำหนด
    if frame_count % process_every_n_frames == 0:
        # ลดขนาดเฟรมเพื่อประมวลผลให้เร็วขึ้น
        scale = 0.7
        small_w = int(original_w * scale)
        small_h = int(original_h * scale)
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        # ตรวจจับวัตถุ
        results = model(small_frame,
                       conf=0.25,
                       iou=0.45,
                       imgsz=416,
                       verbose=False,
                       device='cpu')
        
        # ตรวจสอบว่า YOLO เจออะไรหรือไม่
        if len(results[0].boxes) > 0:
            # YOLO เจอ - ใช้ผลลัพธ์จาก YOLO
            detected_frame = results[0].plot()
            last_detection = cv2.resize(detected_frame, (original_w, original_h))
            yolo_detected = True
        else:
            # YOLO ไม่เจอ - ใช้การตรวจจับแสงสว่าง
            bright_spots, thresh_mask = detect_bright_spots(frame)
            if bright_spots:
                last_detection = draw_bright_detections(frame, bright_spots)
            else:
                last_detection = frame
            yolo_detected = False
    
    # เลือกเฟรมที่จะแสดง
    display_frame = last_detection if last_detection is not None else frame
    
    # คำนวณ FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # แสดงข้อมูลบนหน้าจอ
    cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # แสดงโหมดการทำงาน
    mode_text = "YOLO Mode" if yolo_detected else "Bright Detection Mode"
    mode_color = (0, 255, 0) if yolo_detected else (0, 165, 255)
    cv2.putText(display_frame, mode_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    cv2.putText(display_frame, f'Process every {process_every_n_frames} frames', 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, f'Brightness Threshold: {brightness_threshold}', 
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # แสดงจำนวน detection
    if frame_count % process_every_n_frames == 0:
        if yolo_detected:
            num_detections = len(results[0].boxes)
            cv2.putText(display_frame, f'YOLO Detections: {num_detections}', 
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            bright_spots, _ = detect_bright_spots(frame)
            cv2.putText(display_frame, f'Bright Spots: {len(bright_spots)}', 
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # แสดงภาพหลัก
    cv2.imshow('Drone Detection', display_frame)
    
    # แสดง threshold mask ถ้าต้องการ
    if show_threshold:
        _, thresh_display = detect_bright_spots(frame)
        cv2.imshow('Brightness Threshold', thresh_display)
    
    # จัดการ keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        if process_every_n_frames < 10:
            process_every_n_frames += 1
            print(f"Process every {process_every_n_frames} frames")
    elif key == ord('-'):
        if process_every_n_frames > 1:
            process_every_n_frames -= 1
            print(f"Process every {process_every_n_frames} frames")
    elif key == ord('b'):  # เพิ่ม brightness threshold
        brightness_threshold = min(255, brightness_threshold + 10)
        print(f"Brightness threshold: {brightness_threshold}")
    elif key == ord('v'):  # ลด brightness threshold
        brightness_threshold = max(50, brightness_threshold - 10)
        print(f"Brightness threshold: {brightness_threshold}")
    elif key == ord('d'):  # toggle threshold display
        show_threshold = not show_threshold
        if not show_threshold:
            cv2.destroyWindow('Brightness Threshold')
        print(f"Threshold display: {'ON' if show_threshold else 'OFF'}")
    elif key == ord('s'):  # บันทึกภาพ
        filename = f'detection_{int(time.time())}.jpg'
        cv2.imwrite(filename, display_frame)
        print(f"บันทึกภาพ: {filename}")

# ปิดทุกอย่าง
cap.release()
cv2.destroyAllWindows()
print("จบการทำงาน")
