import cv2
import numpy as np
from typing import List, Dict, Tuple
from config import NightModeConfig

def detect_lights(frame: np.ndarray, config: NightModeConfig) -> List[Dict]:
    """ตรวจจับแสงในภาพสำหรับโหมดกลางคืน"""
    # แปลงเป็นภาพขาวดำ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ทำ blur เพื่อลดนอยซ์
    blurred = cv2.GaussianBlur(gray, (config.blur_size, config.blur_size), 0)
    
    # ตรวจจับพื้นที่สว่าง
    _, thresh = cv2.threshold(blurred, config.light_threshold, 255, cv2.THRESH_BINARY)
    
    # หา contours ของแสง
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < config.min_light_area:
            continue
            
        # หาจุดศูนย์กลางและขนาดของแสง
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w//2
        center_y = y + h//2
        
        # คำนวณความสว่างเฉลี่ยในพื้นที่
        roi = gray[y:y+h, x:x+w]
        brightness = np.mean(roi)
        
        # เพิ่มการตรวจจับในรูปแบบเดียวกับโมเดล AI
        detection = {
            'bbox': (x, y, x+w, y+h),
            'coordinates': {'x': center_x, 'y': center_y},
            'confidence': brightness / 255.0,  # แปลงความสว่างเป็นค่าความมั่นใจ
            'class_name': 'light_source',
            'threat_level': 'YELLOW' if brightness > 200 else 'GREEN'
        }
        detections.append(detection)
    
    return detections

def check_night_mode(frame: np.ndarray, config: NightModeConfig) -> bool:
    """ตรวจสอบว่าควรเปิดโหมดกลางคืนหรือไม่ (สำหรับ auto mode)"""
    if not config.auto_mode:
        return config.enabled
        
    # คำนวณความสว่างเฉลี่ยของภาพ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # สลับโหมดตาม threshold
    return avg_brightness < config.brightness_threshold
