
# ระบบตรวจจับโดรนทางยุทธวิธี (Tactical Drone Detection System)

## คุณสมบัติหลัก
- ตรวจจับโดรน, อากาศยาน, และ UAV อัตโนมัติด้วย AI
- รองรับการทำงานทั้งกลางวันและกลางคืน
- แจ้งเตือนผ่าน LINE ทันทีที่ตรวจพบภัยคุกคาม
- บันทึกภาพและข้อมูลการตรวจจับโดยอัตโนมัติ
- แสดงผลแบบ Real-time พร้อมข้อมูลการวิเคราะห์
- รองรับการควบคุมผ่าน COM Port (สำหรับอุปกรณ์เสริม)

## การติดตั้ง

### ความต้องการของระบบ
- Python 3.9 ขึ้นไป
- OpenCV
- PyTorch
- ultralytics (YOLO)
- Line-bot-sdk

### ขั้นตอนการติดตั้ง
1. ติดตั้ง Python และ pip
2. ติดตั้ง requirements:
```bash
pip install opencv-python torch ultralytics line-bot-sdk
```

### การตั้งค่า LINE Notification
1. สร้าง LINE Bot ที่ [Line Developers Console](https://developers.line.biz/)
2. นำ Channel Secret และ Channel Access Token มาใส่ในไฟล์ config.py
3. เพิ่ม Bot เป็นเพื่อนและส่งข้อความเพื่อรับ User ID
4. นำ User ID มาใส่ในไฟล์ config.py

## การใช้งาน

### เริ่มใช้งานระบบ
```bash
python main.py
```

### พารามิเตอร์ที่รองรับ
```bash
python main.py [options]
  --model, -m        เลือกโมเดล AI (default: best.pt)
  --confidence, -c   ค่าความมั่นใจขั้นต่ำ (default: 0.5)
  --camera          เลือกกล้อง (default: 0)
  --log-level       ระดับการแสดง log (default: INFO)
```

### ปุ่มควบคุม
- **Q** หรือ **ESC**: ออกจากโปรแกรม
- **S**: บันทึกภาพปัจจุบัน
- **R**: รีเซ็ตสถิติ
- **SPACE**: หยุด/เล่น
- **N**: สลับโหมดกลางคืน
- **A**: เปิด/ปิดโหมดกลางคืนอัตโนมัติ
- **F**: สลับโหมดเต็มจอ

## โครงสร้างโปรเจค
- `main.py` - ไฟล์หลักสำหรับรันโปรแกรม
- `config.py` - การตั้งค่าระบบและค่าคงที่ต่างๆ
- `system.py` - ระบบหลักสำหรับการตรวจจับและประมวลผล
- `ui.py` - ส่วนแสดงผลและ UI
- `line_notification.py` - ระบบแจ้งเตือนผ่าน LINE
- `servo_control.py` - ควบคุมอุปกรณ์ผ่าน COM Port

## โครงสร้างไดเรกทอรี
- `detections/` - ภาพที่ตรวจพบโดรนและข้อมูล JSON
- `reports/` - รายงานการปฏิบัติการ
- `logs/` - บันทึกการทำงานของระบบ
- `exports/` - ข้อมูลส่งออก
- `intelligence/` - ข้อมูลวิเคราะห์เชิงลึก

## การแจ้งเตือนผ่าน LINE
ระบบจะส่งการแจ้งเตือนเมื่อ:
- ตรวจพบโดรนที่มีความมั่นใจสูง (>80%)
- ตรวจพบต่อเนื่องอย่างน้อย 3 ครั้ง
- ผ่านไปแล้วอย่างน้อย 5 วินาทีจากการแจ้งเตือนครั้งล่าสุด

ข้อมูลที่ส่ง:
- วันและเวลาที่ตรวจพบ
- ระดับความมั่นใจในการตรวจจับ
- ระดับภัยคุกคาม
- พิกัดที่ตรวจพบ
- ภาพถ่าย ณ จุดที่ตรวจพบ

## การบำรุงรักษา
- ตรวจสอบ log ใน `logs/` เป็นประจำ
- สำรองข้อมูลใน `detections/` และ `reports/` เป็นระยะ
- อัพเดทโมเดล AI เมื่อมีเวอร์ชั่นใหม่

## การแก้ไขปัญหาเบื้องต้น
1. กรณีกล้องไม่ทำงาน:
   - ตรวจสอบการเชื่อมต่อกล้อง
   - ลองเปลี่ยนค่า --camera เป็นค่าอื่น

2. กรณี LINE ไม่แจ้งเตือน:
   - ตรวจสอบ Internet Connection
   - ตรวจสอบ Channel Access Token และ User ID
   - ตรวจสอบว่าได้เพิ่ม Bot เป็นเพื่อนแล้ว

3. กรณี COM Port Error:
   - ตรวจสอบการเชื่อมต่อ
   - ตรวจสอบสิทธิ์การเข้าถึง COM Port


![Detection Example](image/ex1.jpg)
![alt text](image/DIAGRAM.png)



พัฒนาโดย: TEERATHAP YAISUNGNOEN
สถาบัน: King Mongkut's University of Technology Thonburi (KMUTT)
