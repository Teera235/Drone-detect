import cv2
import math
import numpy as np
import json
from typing import Dict, List, Tuple

from config import DisplayConfig, GuidanceConfig, UIState, SOFTWARE_INFO, threat_color_map

# ============== Primitive UI Effects (small, reusable) ==============
class UIEffects:
    @staticmethod
    def glow_rect(img, pt1, pt2, color, thickness=2, glow=5):
        for i in range(glow, 0, -1):
            alpha = 0.3 * (glow - i) / glow
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, (pt1[0]-i, pt1[1]-i), (pt2[0]+i, pt2[1]+i), glow_color, thickness+i)
        cv2.rectangle(img, pt1, pt2, color, thickness)

    @staticmethod
    def gradient(w, h, c1, c2, vertical=True):
        g = np.zeros((h, w, 3), dtype=np.uint8)
        if vertical:
            for y in range(h):
                r = y / h
                g[y, :] = [int(c1[i]*(1-r) + c2[i]*r) for i in range(3)]
        else:
            for x in range(w):
                r = x / w
                g[:, x] = [int(c1[i]*(1-r) + c2[i]*r) for i in range(3)]
        return g

    @staticmethod
    def radar_sweep(img, center, radius, angle, color):
        cv2.circle(img, center, radius, color, 1)
        cv2.circle(img, center, radius//2, color, 1)
        cv2.circle(img, center, radius//4, color, 1)
        ex = int(center[0] + radius * math.cos(angle))
        ey = int(center[1] + radius * math.sin(angle))
        cv2.line(img, center, (ex, ey), color, 2)
        for i in range(8):
            fa = angle - i*0.3
            fa_alpha = 1.0 - i*0.12
            fex = int(center[0] + radius * math.cos(fa))
            fey = int(center[1] + radius * math.sin(fa))
            fc = tuple(int(c * fa_alpha) for c in color)
            cv2.line(img, center, (fex, fey), fc, 1)

effects = UIEffects()

# ============== Guidance helpers (math only) ==============
def roi_rect(width: int, height: int, cfg: GuidanceConfig):
    # centered ROI เมื่อกำหนด roi_size_ratio (0<ratio<=1) ไม่งั้นใช้ margin แบบเดิม
    ratio = getattr(cfg, "roi_size_ratio", 1.0)
    if 0 < ratio < 1.0:
        w = int(width * ratio); h = int(height * ratio)
        cx, cy = width // 2, height // 2
        return (cx - w//2, cy - h//2, cx + w//2, cy + h//2)
    m = int(cfg.roi_margin_ratio * min(width, height))
    return (m, m, width - m, height - m)

def pixel_offset_to_deg(dx_px: float, dy_px: float, width: int, height: int, cfg: GuidanceConfig):
    deg_per_px_h = cfg.fov_deg_h / max(1, width)
    deg_per_px_v = cfg.fov_deg_v / max(1, height)
    left_deg = -dx_px * deg_per_px_h
    up_deg   =  dy_px * deg_per_px_v
    return left_deg, up_deg

def compute_guidance(cx: int, cy: int, width: int, height: int, cfg: GuidanceConfig):
    rx1, ry1, rx2, ry2 = roi_rect(width, height, cfg)
    rcx, rcy = (rx1 + rx2)//2, (ry1 + ry2)//2
    dx, dy_img = cx - rcx, rcy - cy
    left_deg, up_deg = pixel_offset_to_deg(dx, dy_img, width, height, cfg)
    inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
    L = max(1.0, math.hypot(dx, dy_img))
    ex = int(cx - (dx / L) * cfg.line_length_px)
    ey = int(cy + (dy_img / L) * cfg.line_length_px)
    return {"roi": (rx1, ry1, rx2, ry2), "roi_center": (rcx, rcy), "inside": inside,
            "left_deg": left_deg, "up_deg": up_deg, "arrow_end": (ex, ey)}

# ============== Small renderers (each < ~30 lines) ==============
def render_header(overlay, w, display: DisplayConfig, ui: UIState, session: Dict, logo_img):
    h = 100
    hdr = effects.gradient(w, h, display.background_dark, display.panel_dark)
    overlay[0:h, :] = cv2.addWeighted(overlay[0:h, :], 0.3, hdr, 0.7, 0)
    effects.glow_rect(overlay, (0, 0), (w, h), display.primary_orange, 3, 8)
    if logo_img is not None:
        try:
            logo = cv2.resize(logo_img, (60, 60))
            if logo.shape[2] == 4:
                alpha = logo[:,:,3] / 255.0
                for c in range(3):
                    overlay[20:80, 20:80, c] = alpha*logo[:,:,c] + (1-alpha)*overlay[20:80, 20:80, c]
            else:
                overlay[20:80, 20:80] = logo
        except:
            pass
    title = "ORDNANCE ARMY - ENHANCED TACTICAL DRONE DETECTION v3.1"
    cv2.putText(overlay, title, (97, 32), cv2.FONT_HERSHEY_SIMPLEX, display.title_font_scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(overlay, title, (95, 30), cv2.FONT_HERSHEY_SIMPLEX, display.title_font_scale, display.text_white, 2, cv2.LINE_AA)
    ui.pulse_phase += display.pulse_speed
    pulse = 0.7 + 0.3 * math.sin(ui.pulse_phase * 2)
    pc = tuple(int(c * pulse) for c in display.accent_yellow)
    cv2.putText(overlay, "RESTRICTED - ARMY USE ONLY - ENHANCED UI", (95, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pc, 2, cv2.LINE_AA)
    tmap = threat_color_map(display)
    tcol = tmap.get(session.get("threat_level","GREEN"), display.success_green)
    x1,y1,x2,y2 = 95,65,350,85
    _anim_border(overlay,(x1,y1),(x2,y2),tcol,2,ui.frame_count)
    cv2.putText(overlay, f"THREAT LEVEL: {session.get('threat_level','GREEN')}", (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcol, 2, cv2.LINE_AA)
    center = (w - 60, 50)
    ui.radar_angle += 0.1
    effects.radar_sweep(overlay, center, 35, ui.radar_angle, display.accent_yellow)

def _anim_border(img, p1, p2, color, thickness, frame_count):
    w = p2[0]-p1[0]; h = p2[1]-p1[1]; off = int((frame_count*2) % 20); dash = 10
    for x in range(p1[0], p2[0], dash*2):
        if (x + off) % (dash*4) < dash*2:
            cv2.line(img, (x, p1[1]), (min(x+dash, p2[0]), p1[1]), color, thickness)
            cv2.line(img, (x, p2[1]), (min(x+dash, p2[0]), p2[1]), color, thickness)
    for y in range(p1[1], p2[1], dash*2):
        if (y + off) % (dash*4) < dash*2:
            cv2.line(img, (p1[0], y), (p1[0], min(y+dash, p2[1])), color, thickness)
            cv2.line(img, (p2[0], y), (p2[0], min(y+dash, p2[1])), color, thickness)

def render_boxes(overlay, detections, display: DisplayConfig, ui: UIState):
    # ไม่วาด crosshair ติดตามโดรนอีกต่อไป (ให้จุดกลางคงที่เท่านั้น)
    tmap = threat_color_map(display)
    for i, d in enumerate(detections):
        (x1,y1,x2,y2) = d['bbox']; conf = d['confidence']; cname = d['class_name']; level = d['threat_level']
        col = tmap.get(level, display.primary_orange)
        if level == 'RED':
            amp = 0.7 + 0.3 * math.sin(ui.pulse_phase * 4)
            col = tuple(int(c * amp) for c in col)
        effects.glow_rect(overlay, (x1,y1), (x2,y2), col, display.box_thickness, display.glow_intensity if level=='RED' else 5)
        label = f"TGT-{i+1:02d} {cname.upper()} {conf:.1%} [{level}]"
        sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, display.font_scale, 2)[0]
        bg = UIEffects.gradient(sz[0]+30, 40, col, display.panel_dark)
        y0 = max(0, y1-40); x0 = max(0, x1); x1b = min(overlay.shape[1], x0 + bg.shape[1])
        overlay[y0:y0+40, x0:x1b] = cv2.addWeighted(overlay[y0:y0+40, x0:x1b], 0.3, bg[:, :x1b-x0], 0.7, 0)
        cv2.putText(overlay, label, (x0+12, y0+28), cv2.FONT_HERSHEY_SIMPLEX, display.font_scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x0+10, y0+26), cv2.FONT_HERSHEY_SIMPLEX, display.font_scale, display.text_white, 2, cv2.LINE_AA)

def render_center_crosshair(overlay, display: DisplayConfig):
    """เป้ากลาง คงที่ กลางเฟรมเท่านั้น - ไม่ตามโดรน"""
    h, w = overlay.shape[:2]
    cx, cy = w//2, h//2
    
    # วงแหวนเป้า (คงที่)
    cv2.circle(overlay, (cx, cy), 22, display.text_white, 2)
    cv2.circle(overlay, (cx, cy), 40, display.accent_yellow, 2)
    
    # กางเขน (คงที่)
    cv2.line(overlay, (cx-25, cy), (cx+25, cy), display.text_white, 2)
    cv2.line(overlay, (cx, cy-25), (cx, cy+25), display.text_white, 2)
    
    # จุดกลาง (คงที่ - ไม่ใช่สีดำ)
    cv2.circle(overlay, (cx, cy), 3, display.primary_orange, -1)  # เปลี่ยนเป็นสีส้ม
    cv2.circle(overlay, (cx, cy), 5, display.text_white, 1)  # ขอบขาว

def render_guidance(overlay, detections, display: DisplayConfig, guide: GuidanceConfig):
    """แก้ไขการแสดง guidance - จุดดำไม่ควรตามโดรน"""
    h, w = overlay.shape[:2]
    rx1, ry1, rx2, ry2 = roi_rect(w, h, guide)
    cv2.rectangle(overlay, (rx1,ry1), (rx2,ry2), guide.roi_color, 3)
    
    if not detections:
        cv2.putText(overlay, "Outside", (rx1+10, ry1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3, cv2.LINE_AA)
        return
    
    tgt = max(detections, key=lambda d: d["confidence"]) if guide.show_only_highest_conf else detections[0]
    cx, cy = tgt['coordinates']['x'], tgt['coordinates']['y']
    g = compute_guidance(cx, cy, w, h, guide)
    
                    # วาดเส้นชี้ทางจากโดรนไปจุดกลาง
    center = (w//2, h//2)  # จุดกลางจอ
    cv2.line(overlay, (cx, cy), center, guide.vector_color, 3)
    
    # เพิ่ม effect ที่จุดกลาง (ถ้าต้องการ)
    if not g["inside"]:  # ถ้าโดรนอยู่นอกพื้นที่เป้าหมาย
        cv2.circle(overlay, center, 8, guide.vector_color, 2)  # วงกลมสีเขียวรอบจุดกลาง
    
    # ส่งค่าไปยัง COM port
    try:
        # ส่งค่าไปที่ Arduino โดยตรง
        import serial
        ser = serial.Serial("COM5", 9600)
        
        # สร้าง JSON data
        data = {
            "left_deg": round(g["left_deg"], 2),    # องศาซ้าย/ขวา
            "up_deg": round(g["up_deg"], 2),      # องศาขึ้น/ลง
            "is_tracking": True
        }
        
        # แปลงเป็น string และเพิ่ม newline
        json_str = json.dumps(data) + "\n"
        
        # ส่งไปที่ Arduino
        ser.write(json_str.encode())
        ser.flush()  # รอให้ข้อมูลส่งเสร็จ
        
    except Exception as e:
        print(f"[COM] Send failed: {e}")  # แสดง error ถ้าส่งไม่สำเร็จ
    state_text = "Inside" if g["inside"] else "Outside" 
    sc = (0,255,0) if g["inside"] else (0,0,255)
    cv2.putText(overlay, state_text, (rx1+10, ry1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, sc, 4, cv2.LINE_AA)
    
    # แสดงมุม
    left_str = f"Left:{abs(g['left_deg']):.2f}" if g['left_deg'] >= 0 else f"Right:{abs(g['left_deg']):.2f}"
    up_str   = f"Up:{abs(g['up_deg']):.2f}" if g['up_deg'] >= 0 else f"Down:{abs(g['up_deg']):.2f}"
    cv2.putText(overlay, left_str, (rx1+10, ry1+110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, guide.text_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, up_str,   (rx1+10, ry1+145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, guide.text_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, "Av : Target Hit Zone", (rx1, min(h-10, ry2+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

def render_info_panel(overlay, fps, detections, display: DisplayConfig, session: Dict, model_ok: bool, cam_ok: bool):
    h, w = overlay.shape[:2]
    pw, ph = 380, 280
    px, py = w - pw - 15, 100 + 15
    grad = effects.gradient(pw, ph, display.panel_dark, display.background_dark)
    overlay[py:py+ph, px:px+pw] = cv2.addWeighted(overlay[py:py+ph, px:px+pw], 0.2, grad, 0.8, 0)
    effects.glow_rect(overlay, (px,py), (px+pw,py+ph), display.primary_orange, 3, 8)
    cv2.rectangle(overlay, (px+5,py+5), (px+pw-5,py+35), display.primary_orange, -1)
    cv2.putText(overlay, "ENHANCED TACTICAL INTELLIGENCE", (px+15, py+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display.text_white, 2, cv2.LINE_AA)
    tmap = threat_color_map(display)
    items = [
        ("Performance:", f"{fps:.1f} FPS"),
        ("Active Targets:", f"{len(detections)}"),
        ("Total Contacts:", f"{session.get('detection_count',0):,}"),
        ("Frames Analyzed:", f"{session.get('total_frames',0):,}"),
        ("Threat Status:", session.get("threat_level","GREEN")),
        ("Operation ID:", session.get("session_id","")[:12] + "..."),
    ]
    for i,(k,v) in enumerate(items):
        y = py + 55 + i*25
        cv2.putText(overlay, k, (px+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display.accent_yellow, 1, cv2.LINE_AA)
        col = display.text_white
        if k=="Threat Status:": col = tmap.get(v, display.text_white)
        elif k=="Performance:" and fps < 15: col = display.danger_red
        elif k=="Active Targets:" and len(detections) > 0: col = display.warning_amber
        cv2.putText(overlay, v, (px+150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    sy = py + ph - 40
    stats = [("AI MODEL", display.success_green if model_ok else display.danger_red),
             ("CAMERA", display.success_green if cam_ok else display.danger_red),
             ("RECORDING", display.warning_amber)]
    for i,(name,color) in enumerate(stats):
        xo = i * 120
        cx = px + 30 + xo
        cv2.circle(overlay, (cx, sy+10), 8, color, -1)
        cv2.circle(overlay, (cx, sy+10), 8, display.text_white, 1)
        cv2.putText(overlay, name, (cx+15, sy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display.text_light, 1, cv2.LINE_AA)

def render_radar_footer_controls(overlay, display: DisplayConfig, ui: UIState, detections: List[Dict]):
    h, w = overlay.shape[:2]

    # ---------- Mini radar (bottom-left) ----------
    size = 150
    x = 20
    y = h - size - 80
    center = (x + size // 2, y + size // 2)
    radius = size // 2 - 10

    # Base disk
    cv2.circle(overlay, center, size // 2, display.panel_dark, -1)

    # Sweep first
    ui.radar_angle += 0.1
    effects.radar_sweep(overlay, center, radius, ui.radar_angle, display.accent_yellow)

    # Plot detections AFTER sweep so they stay on top
    tmap = threat_color_map(display)
    
    # จุดดำ (Center) อยู่ตรงกลางเรดาร์เสมอ
    cv2.circle(overlay, center, 4, (0, 0, 0), -1)  # จุดดำ
    cv2.circle(overlay, center, 6, display.text_white, 2)  # ขอบขาว
    
    for i, d in enumerate(detections):
        if 'coordinates' not in d:
            continue
            
        dx = d['coordinates']['x']
        dy = d['coordinates']['y']
        
        # คำนวณตำแหน่งในเรดาร์ (ปรับ scale ให้เหมาะสม)
        scale_factor = 0.6  # ลดลงเพื่อให้จุดไม่ออกนอกเรดาร์ง่าย
        rx = int(center[0] + (dx - w/2) * (radius * scale_factor) / (w/2))
        ry = int(center[1] + (dy - h/2) * (radius * scale_factor) / (h/2))
        
        # ตรวจสอบว่าอยู่ในขอบเขตเรดาร์หรือไม่
        distance_from_center = math.sqrt((rx - center[0])**2 + (ry - center[1])**2)
        
        if distance_from_center <= radius:
            # กำหนดสีตาม threat level (default = RED)
            threat_level = d.get("threat_level", "RED")
            col = tmap.get(threat_level, display.danger_red)
            
            # วาดจุดเป้าหมาย (โดรน)
            cv2.circle(overlay, (rx, ry), 6, col, -1)  # จุดสีแดง
            cv2.circle(overlay, (rx, ry), 8, display.text_white, 2)  # ขอบขาว
            
            # เพิ่ม glow effect สำหรับ RED threat
            if threat_level == "RED":
                alpha = 0.7 + 0.3 * math.sin(ui.pulse_phase * 4)  # Effect กระพริบ
                glow_col = tuple(int(c * alpha) for c in col)
                cv2.circle(overlay, (rx, ry), 12, glow_col, 1)
                cv2.circle(overlay, (rx, ry), 16, glow_col, 1)

    # เพิ่ม label ให้เรดาร์
    cv2.putText(overlay, "RADAR", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display.text_white, 1, cv2.LINE_AA)

    # ---------- Footer + controls ----------
    fh = 70; fy = h - fh
    grad = effects.gradient(w, fh, display.background_dark, display.panel_dark)
    overlay[fy:h, :] = cv2.addWeighted(overlay[fy:h, :], 0.3, grad, 0.7, 0)
    cv2.line(overlay, (0, fy), (w, fy), display.primary_orange, 3)

    txt = f"© 2025 {SOFTWARE_INFO['organization']} | ENHANCED UI | {SOFTWARE_INFO['classification']}"
    cv2.putText(overlay, txt, (22, fy + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, txt, (20, fy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display.text_white, 2, cv2.LINE_AA)
    contact = f"Contact: {SOFTWARE_INFO['contact']} | Developer: {SOFTWARE_INFO['developer']} (KMUTT)"
    cv2.putText(overlay, contact, (20, fy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display.accent_yellow, 1, cv2.LINE_AA)

    controls = "[Q]uit | [S]ave | [R]eset | [SPACE]Pause | [ESC]Exit"
    cx = w - len(controls) * 6 - 20
    cv2.rectangle(overlay, (cx - 10, fy + 5), (w - 10, fy + 20), display.panel_dark, -1)
    cv2.putText(overlay, controls, (cx, fy + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display.text_light, 1, cv2.LINE_AA)

# ============== Public entry: render_overlay ==============
def render_overlay(frame, detections, fps, display: DisplayConfig, guide: GuidanceConfig,
                   ui: UIState, session: Dict, logo_img, model_ok: bool, cam_ok: bool):
    ui.frame_count += 1
    overlay = frame.copy()
    render_header(overlay, frame.shape[1], display, ui, session, logo_img)
    render_boxes(overlay, detections, display, ui)
    render_center_crosshair(overlay, display)        # เป้าคงที่กลางจอ - แก้ไขแล้ว
    render_guidance(overlay, detections, display, guide)  # แก้ไขแล้ว
    render_info_panel(overlay, fps, detections, display, session, model_ok, cam_ok)
    render_radar_footer_controls(overlay, display, ui, detections)  # แก้ไขแล้ว
    return overlay