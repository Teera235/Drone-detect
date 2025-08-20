import serial
import time
import json

# เปิด Serial Port COM5 ที่ baudrate 9600
try:
    ser = serial.Serial("COM5", 9600)
    time.sleep(2) # รอให้ Arduino พร้อม
    print("Connected to Arduino on COM5")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

while True:
    try:
        # รับค่ามุมจากผู้ใช้
        angle = input("Enter angle (0-180): ")
        
        # ตรวจสอบว่าเป็นตัวเลขและอยู่ในช่วง 0-180
        angle = int(angle)
        if not (0 <= angle <= 180):
            print("Angle must be between 0 and 180")
            continue
            
        # สร้าง JSON data
        data = {
            "servo": angle
        }
        
        # แปลงเป็น string และเพิ่ม newline
        json_str = json.dumps(data) + "\n"
        
        # ส่งไปที่ Arduino
        ser.write(json_str.encode())
        print(f"Sent: {json_str.strip()}")
        
        # รอการตอบกลับ
        if ser.in_waiting:
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")
            
    except ValueError:
        print("Please enter a valid number")
    except Exception as e:
        print(f"Error: {e}")
        break

# ปิด Serial port เมื่อจบโปรแกรม
ser.close()
