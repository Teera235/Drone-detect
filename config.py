
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

# ==================== SOFTWARE INFORMATION ====================
SOFTWARE_INFO = {
    "name": "ORDNANCE Army Drone Detection System",
    "version": "1.0.0",
    "copyright": "Copyright (c) 2025 ORDNANCE ARMY DIVISION. All Rights Reserved.",
    "developer": "TEERATHAP YAISUNGNOEN",
    "institution": "King Mongkut's University of Technology Thonburi (KMUTT)",
    "contact": "teerathap.nist@mail.kmutt.ac.th",
    "organization": "ORDNANCE ARMY DIVISION",
    "classification": "RESTRICTED - ARMY USE ONLY"
}

# LINE Messaging API Configuration
LINE_CONFIG = {
    "channel_secret": "eef7e611633920ea0aeca952a53e7be0",
    "channel_access_token": "u7imZTciWkSxmouvjgW3cMYLg5yrxiGwlOxWu+hWwGmzxZPoMAxROhk8EAmOTaD+O96icERaC+Yn3+qCuv4tz7cwlAV3s9ZWEU5XTr6CxAwRKyt9Yn24hJPjmr90TZIE1HNzZpzfEwoj1yfQvf5wSgdB04t89/1O/w1cDnyilFU=",
    "notification_enabled": True,  # เปิด/ปิดการแจ้งเตือน
    "user_id": "Ue88385473bf80d2e0639b89e4f821a70"  # จะถูกอัพเดทเมื่อมีการแชทกับบอทครั้งแรก
}

# ==================== Configuration ====================
@dataclass
class NightModeConfig:
    enabled: bool = False
    auto_mode: bool = True
    brightness_threshold: int = 50  # ค่าความสว่างที่จะสลับเป็นโหมดกลางคืน (0-255)
    light_threshold: int = 200  # ค่าความสว่างขั้นต่ำในการตรวจจับแสง
    min_light_area: int = 50  # พื้นที่ขั้นต่ำของแสงที่จะตรวจจับ (pixels)
    blur_size: int = 15  # ขนาด blur สำหรับลดนอยซ์
    detection_color: Tuple[int,int,int] = (0,255,255)  # สีของการตรวจจับแสง (BGR)

@dataclass
class DetectionConfig:
    model_path: str = "best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 300
    target_classes: List[str] = None
    auto_save_detections: bool = True
    save_directory: str = "detections"
    night_mode: NightModeConfig = None
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = ["drone", "aircraft", "helicopter", "uav"]
        if self.night_mode is None:
            self.night_mode = NightModeConfig()

@dataclass
class DisplayConfig:
    window_name: str = "ORDNANCE ARMY - Tactical Drone Detection System v3.1"
    # Palette (BGR)
    primary_orange: Tuple[int,int,int] = (0,140,255)
    secondary_orange: Tuple[int,int,int] = (0,100,200)
    accent_yellow: Tuple[int,int,int] = (0,255,255)
    warning_amber: Tuple[int,int,int] = (0,200,255)
    success_green: Tuple[int,int,int] = (0,255,100)
    danger_red: Tuple[int,int,int] = (0,50,255)
    background_dark: Tuple[int,int,int] = (25,25,25)
    panel_dark: Tuple[int,int,int] = (40,40,40)
    text_white: Tuple[int,int,int] = (255,255,255)
    text_light: Tuple[int,int,int] = (200,200,200)
    border_color: Tuple[int,int,int] = (100,100,100)
    # Legacy aliases
    primary_color: Tuple[int,int,int] = (0,140,255)
    secondary_color: Tuple[int,int,int] = (255,255,255)
    accent_color: Tuple[int,int,int] = (0,255,255)
    alert_color: Tuple[int,int,int] = (0,50,255)
    army_color: Tuple[int,int,int] = (0,140,255)
    military_color: Tuple[int,int,int] = (0,140,255)
    text_color: Tuple[int,int,int] = (0,0,0)
    # Styling
    box_thickness: int = 3
    font_scale: float = 0.6
    credit_font_scale: float = 0.7
    header_font_scale: float = 0.8
    title_font_scale: float = 1.0
    # Animation
    pulse_speed: float = 0.05
    glow_intensity: int = 15

@dataclass
class GuidanceConfig:
    fov_deg_h: float = 62.2
    roi_size_ratio: float = 0.6  #0<value<=1; e.g., 0.5 = half-size ROI
    fov_deg_v: float = 48.8
    roi_margin_ratio: float = 0.08
    line_length_px: int = 120
    show_only_highest_conf: bool = True
    roi_color: Tuple[int,int,int] = (255,0,255)
    vector_color: Tuple[int,int,int] = (0,255,0)
    text_color: Tuple[int,int,int] = (0,255,0)

@dataclass
class UIState:
    frame_count: int = 0
    radar_angle: float = 0.0
    pulse_phase: float = 0.0

def threat_color_map(display: DisplayConfig):
    return {
        "GREEN": display.success_green,
        "YELLOW": display.warning_amber,
        "RED": display.danger_red,
    }
