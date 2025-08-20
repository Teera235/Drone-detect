import serial
import time
from typing import Tuple
import logging
import json

class COMController:
    def __init__(self, port='COM5', baud_rate=9600):
        """
        ตั้งค่าการเชื่อมต่อ Serial port
        port: COM port ที่ต้องการเชื่อมต่อ
        baud_rate: ความเร็วในการสื่อสาร
        """
        self.logger = logging.getLogger("ORDNANCE_DRONE_DETECTION")
        self.logger.setLevel(logging.DEBUG)  # เพิ่ม debug level
        
        # เพิ่ม handler ถ้ายังไม่มี
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.serial = None
        self.connected = False
        
        self.logger.debug(f"Attempting to connect to {port} at {baud_rate} baud...")
        
        try:
            # ตรวจสอบว่า port มีอยู่จริง
            import serial.tools.list_ports
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            self.logger.debug(f"Available COM ports: {available_ports}")
            
            if port not in available_ports:
                raise Exception(f"Port {port} not found. Available ports: {available_ports}")
            
            self.serial = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # รอการเชื่อมต่อ
            
            if not self.serial.is_open:
                self.serial.open()
            
            # ล้าง buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # ทดสอบการเชื่อมต่อ
            test_data = {"test": True}
            self.logger.debug(f"Sending test data: {test_data}")
            self.serial.write(json.dumps(test_data).encode() + b'\n')
            
            # รอและอ่านการตอบกลับ
            response = self.serial.readline().decode().strip()
            self.logger.debug(f"Test response: {response}")
            
            self.connected = True
            self.logger.info(f"Successfully connected to {port}")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Connection failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # ให้ error ขึ้นไปที่โปรแกรมหลัก
    
    def send_tracking_data(self, left_deg: float, up_deg: float, is_tracking: bool = True):
        """
        ส่งข้อมูลการติดตามผ่าน COM port
        left_deg: องศาซ้าย/ขวา (- คือซ้าย, + คือขวา)
        up_deg: องศาขึ้น/ลง (+ คือขึ้น, - คือลง)
        is_tracking: สถานะการติดตาม (True = กำลังติดตามโดรน, False = ไม่พบโดรน)
        """
        if not self.connected:
            print("Not connected to COM port")
            return
            
        try:
            # สร้าง JSON data ส่งค่าองศาไปตรงๆ
            data = {
                "left_deg": round(left_deg, 2),  # องศาซ้าย/ขวา
                "up_deg": round(up_deg, 2),      # องศาขึ้น/ลง
                "is_tracking": is_tracking
            }
            
            # แปลงเป็น string และเพิ่ม newline
            json_str = json.dumps(data) + "\n"
            
            # ส่งข้อมูล
            self.serial.write(json_str.encode())
            self.serial.flush()  # รอให้ข้อมูลส่งเสร็จ
            
            # รอและอ่านการตอบกลับ
            if self.serial.in_waiting:
                response = self.serial.readline().decode().strip()
                self.logger.debug(f"Arduino response: {response}")
                
            self.logger.debug(f"[COM] Sent: {json_str.strip()}")
        except Exception as e:
            self.logger.error(f"[COM] Send failed: {e}")
    
    def close(self):
        """ปิดการเชื่อมต่อ"""
        if self.serial:
            # ส่งข้อมูลว่าไม่มีการติดตาม
            self.send_tracking_data(0, 0, False)
            time.sleep(0.1)  # รอให้ข้อมูลส่งเสร็จ
            
            self.serial.close()
            self.connected = False
            self.logger.info("[COM] Connection closed")
