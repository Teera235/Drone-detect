import cv2
import time
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
process_every_n_frames = 3  # ประมวลผลทุก 3 เฟรม
last_detection = None

print("เริ่มการตรวจจับ (CPU mode)...")
print("กด 'q' เพื่อออก | กด '+' เพื่อลด frame skip | กด '-' เพื่อเพิ่ม frame skip")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    original_h, original_w = frame.shape[:2]
    
    # ประมวลผล YOLO เฉพาะเฟรมที่กำหนด
    if frame_count % process_every_n_frames == 0:
        # ลดขนาดเฟรมเพื่อประมวลผลให้เร็วขึ้น
        scale = 0.7  # ลดเหลือ 70%
        small_w = int(original_w * scale)
        small_h = int(original_h * scale)
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        # ตรวจจับวัตถุ (ใช้การตั้งค่าเพื่อความเร็ว)
        results = model(small_frame,
                       conf=0.1,      # confidence threshold
                       iou=0.1,       # IoU threshold  
                       imgsz=416,      # ขนาด input ที่เล็กลง
                       verbose=False,   # ปิด debug messages
                       device='cpu')    # บังคับใช้ CPU
        
        # วาดผลลัพธ์และขยายกลับเป็นขนาดเดิม
        detected_frame = results[0].plot()
        last_detection = cv2.resize(detected_frame, (original_w, original_h))
    
    # เลือกเฟรมที่จะแสดง
    display_frame = last_detection if last_detection is not None else frame
    
    # คำนวณ FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # แสดงข้อมูลบนหน้าจอ
    cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f'CPU Mode | Process every {process_every_n_frames} frames', 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # แสดงจำนวน detection (ถ้ามี)
    if last_detection is not None and frame_count % process_every_n_frames == 0:
        num_detections = len(results[0].boxes) if len(results[0].boxes) > 0 else 0
        cv2.putText(display_frame, f'Detections: {num_detections}', 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # แสดงภาพ
    cv2.imshow('Drone Detection', display_frame)
    
    # จัดการ keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):  # เพิ่มความเร็ว (ลด frame processing)
        if process_every_n_frames < 10:
            process_every_n_frames += 1
            print(f"Process every {process_every_n_frames} frames (เร็วขึ้น)")
    elif key == ord('-'):  # ลดความเร็ว (เพิ่ม frame processing)
        if process_every_n_frames > 1:
            process_every_n_frames -= 1
            print(f"Process every {process_every_n_frames} frames (ช้าลง แต่แม่นขึ้น)")
    elif key == ord('s'):  # บันทึกภาพ
        filename = f'detection_{int(time.time())}.jpg'
        cv2.imwrite(filename, display_frame)
        print(f"บันทึกภาพ: {filename}")

# ปิดทุกอย่าง
cap.release()
cv2.destroyAllWindows()
print("จบการทำงาน")
