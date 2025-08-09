import cv2
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกแล้วจากไฟล์ .pt
model = YOLO('best.pt')

# เปิดวีดีโอ (สามารถเลือกไฟล์ได้)
video_path = "videoplayback.mp4"  # ใส่ p
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุในภาพ (frame) ด้วยโมเดล YOLO
    results = model(frame)
    
    # แสดงผลลัพธ์บนภาพ
    frame_with_results = results[0].plot()  # plot() เป็นการเพิ่ม box และ labels ลงในภาพ

    # แสดงภาพที่ตรวจจับ
    cv2.imshow('Drone Detection', frame_with_results)

    # กด 'q' เพื่อออกจากการแสดงผล
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดวีดีโอและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()
