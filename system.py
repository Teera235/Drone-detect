
import os
import cv2
import json
import time
import math
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from queue import Queue

from ultralytics import YOLO

from config import (DetectionConfig, DisplayConfig, GuidanceConfig, 
                   UIState, SOFTWARE_INFO, threat_color_map, LINE_CONFIG)
from ui import render_overlay

# ==================== Logging ====================
import logging

class MilitaryLogger:
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("ORDNANCE_DRONE_DETECTION")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/ordnance_detection_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        fmt = logging.Formatter("%(asctime)s | ORDNANCE | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        if not self.logger.handlers:
            self.logger.addHandler(fh); self.logger.addHandler(ch)
    def get(self): return self.logger

# ==================== Core System ====================
class OrdnanceDroneDetectionSystem:
    def __init__(self, det: DetectionConfig, disp: DisplayConfig, guide: GuidanceConfig, log_level="INFO"):
        self.detection_config = det
        self.display_config = disp
        self.guidance_config = guide
        self.logger = MilitaryLogger(log_level).get()
        self.model = None
        self.cap = None
        self.logo = None
        self.ui = UIState()
        self.fps_counter = 0
        self.fps_start = time.time()
        
        # สร้าง queue สำหรับส่งข้อความ LINE
        self.line_queue = Queue()
        self.line_thread = None
        self.last_line_sent = 0  # เวลาที่ส่งข้อความ LINE ล่าสุด
        self.min_line_interval = 5.0  # ระยะเวลาขั้นต่ำระหว่างการส่งข้อความ LINE (วินาที)
        self.consecutive_detections = 0  # นับจำนวนครั้งที่ตรวจจับต่อเนื่อง
        self.min_consecutive_detections = 3  # จำนวนครั้งขั้นต่ำที่ต้องตรวจจับได้ก่อนส่งข้อความ
        
        # เพิ่ม COM Controller
        from servo_control import COMController
        try:
            self.com_controller = COMController()
        except Exception as e:
            self.logger.error(f"[COM] Init failed: {e}")
            self.com_controller = None
            
        # เพิ่ม Line Notifier
        from line_notification import LineNotifier
        try:
            if LINE_CONFIG["notification_enabled"]:
                self.line_notifier = LineNotifier(LINE_CONFIG["channel_access_token"])
                self.logger.info("[LINE] Notification system initialized")
                # เริ่ม thread สำหรับส่งข้อความ LINE
                self.line_thread = threading.Thread(target=self._line_worker, daemon=True)
                self.line_thread.start()
            else:
                self.line_notifier = None
                self.logger.info("[LINE] Notification system disabled")
        except Exception as e:
            self.logger.error(f"[LINE] Init failed: {e}")
            self.line_notifier = None
            
        self.session = {
            "start_time": time.time(),
            "total_frames": 0,
            "detection_count": 0,
            "saved_detections": 0,
            "avg_fps": 0.0,
            "peak_fps": 0.0,
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "threat_level": "GREEN",
        }
        self._startup_banner()
        self._prepare_dirs()
        self._load_logo()

    # -------- setup --------
    def _startup_banner(self):
        self.logger.info("="*80)
        self.logger.info(f"{SOFTWARE_INFO['name']} v{SOFTWARE_INFO['version']}")
        self.logger.info(f"{SOFTWARE_INFO['copyright']}")
        self.logger.info(f"Developer: {SOFTWARE_INFO['developer']} from {SOFTWARE_INFO['institution']}")
        self.logger.info(f"{SOFTWARE_INFO['classification']}")
        self.logger.info(f"Operation ID: {self.session['session_id']}")
        self.logger.info(f"OpenCV Version: {cv2.__version__}")
        self.logger.info("System init...")

    def _prepare_dirs(self):
        for d in [self.detection_config.save_directory, "logs", "reports", "exports", "intelligence"]:
            os.makedirs(d, exist_ok=True)

    def _load_logo(self):
        p = "logo.png"
        if os.path.exists(p):
            try:
                self.logo = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if self.logo is not None:
                    self.logo = cv2.resize(self.logo, (60, 60))
                    self.logger.info(f"[OK] Logo loaded: {p}")
                else:
                    self.logger.warning(f"[WARN] Logo not readable: {p}")
            except Exception as e:
                self.logger.warning(f"[WARN] Logo error: {e}")
        else:
            self.logger.info("[INFO] No logo file; using text branding.")

    # -------- model/camera --------
    def initialize_model(self) -> bool:
        try:
            mp = Path(self.detection_config.model_path)
            if not mp.exists():
                raise FileNotFoundError(f"Model file not found: {mp}")
            self.logger.info("Loading YOLO model...")
            self.model = YOLO(str(mp))
            self.logger.info(f"[OK] Model: {mp} ({mp.stat().st_size/(1024*1024):.2f} MB)")
            self.logger.info(f"Classes: {list(self.model.names.values())}")
            self.logger.info(f"Targets: {self.detection_config.target_classes}")
            return True
        except Exception as e:
            self.logger.error(f"[ERROR] Load model failed: {e}")
            return False

    def initialize_camera(self, cam_id: int = 0) -> bool:
        try:
            self.logger.info(f"Opening camera {cam_id}...")
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(cam_id, backend)
                    if cap.isOpened():
                        self.cap = cap; break
                except: pass
            if not self.cap or not self.cap.isOpened():
                for alt in [1, 2, -1]:
                    cap = cv2.VideoCapture(alt)
                    if cap.isOpened(): self.cap = cap; break
            if not self.cap or not self.cap.isOpened():
                raise IOError("Cannot access camera")
            for prop, val in [(cv2.CAP_PROP_FRAME_WIDTH,1280),(cv2.CAP_PROP_FRAME_HEIGHT,720),
                              (cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,1)]:
                try: self.cap.set(prop, val)
                except: pass
            ok, frm = self.cap.read()
            if not ok or frm is None: raise IOError("Camera opened but no frames")
            w,h,fps = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"[OK] Camera {w}x{h} @ {fps:.1f} FPS")
            return True
        except Exception as e:
            self.logger.error(f"[ERROR] Camera init failed: {e}")
            self.logger.error("Troubleshoot: permissions; in-use; run as admin")
            return False

    # -------- detection & stats --------
    def _classify_threat(self, cname: str, conf: float) -> str:
        if conf > 0.8: return "RED"
        if conf > 0.6: return "YELLOW"
        return "GREEN"

    def process_detections(self, results) -> List[Dict]:
        out = []
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                conf = float(b.conf.item())
                if conf < self.detection_config.confidence_threshold: continue
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls.item()); cname = self.model.names[cls_id]
                if cname.lower() not in [c.lower() for c in self.detection_config.target_classes]:
                    continue
                out.append({
                    "bbox": (x1,y1,x2,y2),
                    "confidence": conf,
                    "class_name": cname,
                    "class_id": cls_id,
                    "timestamp": datetime.now().isoformat(),
                    "area": (x2-x1)*(y2-y1),
                    "threat_level": self._classify_threat(cname, conf),
                    "coordinates": {"x": (x1+x2)//2, "y": (y1+y2)//2},
                })
        self.session["threat_level"] = max([d["threat_level"] for d in out], default="GREEN")
        return out

    def calc_fps(self) -> float:
        self.fps_counter += 1
        el = time.time() - self.fps_start
        if el >= 1.0:
            fps = self.fps_counter / el
            self.fps_counter = 0; self.fps_start = time.time()
            if self.session["avg_fps"] == 0: self.session["avg_fps"] = fps
            else: self.session["avg_fps"] = 0.8*self.session["avg_fps"] + 0.2*fps
            if fps > self.session["peak_fps"]: self.session["peak_fps"] = fps
            return fps
        return self.session["avg_fps"]

    def save_detection_frame(self, frame, detections) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fn = f"{self.detection_config.save_directory}/enhanced_tactical_detection_{ts}.jpg"
        cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta = {
            "filename": fn, "timestamp": datetime.now().isoformat(),
            "operation_id": self.session["session_id"], "threat_level": self.session["threat_level"],
            "detections": detections, "classification": SOFTWARE_INFO["classification"],
            "software_info": SOFTWARE_INFO, "enhanced_ui": True, "version": SOFTWARE_INFO["version"],
        }
        with open(fn.replace(".jpg","_enhanced_tactical_data.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        # ส่งข้อมูลไปที่ queue เพื่อส่ง LINE notification
        if self.line_notifier and LINE_CONFIG["user_id"]:
            current_time = time.time()
            
            if detections:
                target = max(detections, key=lambda d: d["confidence"])
                # เพิ่มจำนวนครั้งที่ตรวจจับได้ต่อเนื่อง ถ้าความมั่นใจมากกว่า 80%
                if target["confidence"] > 0.20:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0
            else:
                self.consecutive_detections = 0
            
            # ส่งข้อความถ้า:
            # 1. ตรวจจับได้ต่อเนื่องตามจำนวนครั้งที่กำหนด
            # 2. ผ่านไปนานพอแล้วหลังจากส่งครั้งล่าสุด
            # 3. มีการตรวจจับที่มั่นใจมากกว่า 80%
            if (self.consecutive_detections >= self.min_consecutive_detections and
                current_time - self.last_line_sent >= self.min_line_interval and
                detections):
                
                target = max(detections, key=lambda d: d["confidence"])
                if target["confidence"] > 0.80:
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message = (f"⚠️ พบโดรน!\n"
                            f"เวลา: {current_datetime}\n"
                            f"ความมั่นใจ: {target['confidence']:.1%}\n"
                            f"ระดับภัยคุกคาม: {target['threat_level']}\n"
                            f"พิกัด: ({target['coordinates']['x']}, {target['coordinates']['y']})")
                    self.line_queue.put((fn, message))
                    self.last_line_sent = current_time
                    self.consecutive_detections = 0  # รีเซ็ตหลังจากส่งข้อความ
            
        self.session["saved_detections"] += 1
        return fn

    def _line_worker(self):
        """Thread worker สำหรับส่งข้อความ LINE"""
        while True:
            try:
                # รอรับข้อมูลจาก queue
                image_path, message = self.line_queue.get()
                if image_path is None:  # signal to stop
                    break
                    
                # ส่งข้อความไป LINE
                try:
                    self.line_notifier.send_image(
                        LINE_CONFIG["user_id"],
                        image_path,
                        message
                    )
                except Exception as e:
                    self.logger.error(f"[LINE] Send failed: {e}")
                    
                # บอก queue ว่าทำงานเสร็จแล้ว
                self.line_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"[LINE] Worker error: {e}")
                
    def _reset_stats(self):
        self.session.update({
            "start_time": time.time(), "total_frames": 0, "detection_count": 0, "saved_detections": 0,
            "avg_fps": 0.0, "peak_fps": 0.0, "threat_level": "GREEN"
        })
        self.ui = UIState()
        self.logger.info("[STATS] Reset")

    # -------- main loop --------
    def run(self):
        if not self.initialize_model(): return
        if not self.initialize_camera(): return
        self.logger.info("[DEPLOY] Controls: [Q] [S] [R] [SPACE] [N] [A] [ESC]")
        self.logger.info("[MODE] N = Toggle Night Mode, A = Auto Night Mode")
        paused = False
        
        # Import night detection module
        from night_detection import detect_lights, check_night_mode
        try:
            while True:
                if not paused:
                    ok, frame = self.cap.read()
                    if not ok or frame is None: self.logger.warning("Capture fail"); break
                    self.session["total_frames"] += 1

                    # ตรวจสอบโหมดกลางคืน
                    night_mode = check_night_mode(frame, self.detection_config.night_mode)

                    # เลือกวิธีตรวจจับตามโหมด
                    if night_mode:
                        # โหมดกลางคืน - ตรวจจับแสง
                        dets = detect_lights(frame, self.detection_config.night_mode)
                        self.session["mode"] = "NIGHT"
                    else:
                        # โหมดกลางวัน - ใช้ YOLO model
                        results = self.model(frame, conf=self.detection_config.confidence_threshold,
                                           iou=self.detection_config.iou_threshold,
                                           max_det=self.detection_config.max_detections,
                                           verbose=False)
                        dets = self.process_detections(results)
                        self.session["mode"] = "DAY"

                    self.session["detection_count"] += len(dets)
                    for d in dets:
                        if d["threat_level"] in ["RED","YELLOW"]:
                            self.logger.warning(f"[THREAT] {d['threat_level']} {d['class_name']} {d['confidence']:.1%} at ({d['coordinates']['x']},{d['coordinates']['y']})")
                    fps = self.calc_fps()
                    # ส่ง com_controller ไปให้ guidance
                    self.guidance_config.com_controller = self.com_controller
                    
                    frame = render_overlay(frame, dets, fps, self.display_config, self.guidance_config,
                                           self.ui, self.session, self.logo,
                                           model_ok=self.model is not None, cam_ok=self.cap is not None and self.cap.isOpened())
                    if dets and self.detection_config.auto_save_detections:
                        fn = self.save_detection_frame(frame, dets)
                        self.logger.info(f"[SAVE] Auto intelligence: {fn}")
                cv2.imshow(self.display_config.window_name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27): break
                elif k == ord('s'):
                    if 'dets' in locals(): self.logger.info(f"[SAVE] Manual: {self.save_detection_frame(frame, dets)}")
                elif k == ord('r'): self._reset_stats()
                elif k == ord(' '): paused = not paused; self.logger.info(f"[STATE] {'PAUSED' if paused else 'ACTIVE'}")
                elif k == ord('n'):  # สลับโหมดกลางคืนด้วยตัวเอง
                    self.detection_config.night_mode.enabled = not self.detection_config.night_mode.enabled
                    self.detection_config.night_mode.auto_mode = False
                    self.logger.info(f"[MODE] Night Mode: {'ON' if self.detection_config.night_mode.enabled else 'OFF'}")
                elif k == ord('a'):  # เปิด/ปิดโหมดอัตโนมัติ
                    self.detection_config.night_mode.auto_mode = not self.detection_config.night_mode.auto_mode
                    self.logger.info(f"[MODE] Auto Night Mode: {'ON' if self.detection_config.night_mode.auto_mode else 'OFF'}")
                elif k == ord('f'):
                    cv2.setWindowProperty(self.display_config.window_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty(self.display_config.window_name, cv2.WND_PROP_FULLSCREEN)==0 else cv2.WINDOW_NORMAL)
        except KeyboardInterrupt:
            self.logger.info("[STOP] Operator terminated")
        except Exception as e:
            self.logger.error(f"[ERROR] {e}")
        finally:
            self.cleanup()

    # -------- teardown/report --------
    def _report(self) -> Dict:
        run = time.time() - self.session["start_time"]
        
        # แปลง config เป็น dict ก่อน
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            return obj
            
        return {
            "mission_info": {
                "operation_id": self.session["session_id"],
                "software": SOFTWARE_INFO,
                "classification": SOFTWARE_INFO["classification"],
                "ui_version": "Enhanced v3.1",
                "start_time": datetime.fromtimestamp(self.session["start_time"]).isoformat(),
                "end_time": datetime.now().isoformat(),
                "runtime_seconds": run,
                "runtime_formatted": f"{run//3600:02.0f}:{(run%3600)//60:02.0f}:{run%60:02.0f}",
            },
            "metrics": {
                "total_frames": self.session["total_frames"],
                "average_fps": (self.session["total_frames"]/run) if run>0 else 0,
                "peak_fps": self.session["peak_fps"],
                "smoothed_avg_fps": self.session["avg_fps"],
            },
            "tactical": {
                "total_detections": self.session["detection_count"],
                "saved_intelligence": self.session["saved_detections"],
                "detection_rate": (self.session["detection_count"]/max(1,self.session['total_frames'])),
                "final_threat_level": self.session["threat_level"],
            },
            "configuration": {
                "detection_config": to_dict(self.detection_config),
                "display_config": to_dict(self.display_config),
            }
        }

    def cleanup(self):
        if self.cap is not None: self.cap.release()
        if self.com_controller is not None: self.com_controller.close()
        # ปิด LINE worker thread
        if self.line_thread is not None:
            self.line_queue.put((None, None))  # signal thread to stop
            self.line_thread.join(timeout=5)  # รอให้ thread จบการทำงาน
        cv2.destroyAllWindows()
        report = self._report()
        rf = f"reports/enhanced_mission_report_{self.session['session_id']}.json"
        with open(rf, "w", encoding="utf-8") as f: json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info("="*80)
        self.logger.info("[MISSION COMPLETE]")
        self.logger.info(f"Frames: {report['metrics']['total_frames']} | AvgFPS: {report['metrics']['average_fps']:.2f} | Peak: {report['metrics']['peak_fps']:.2f}")
        self.logger.info(f"Contacts: {report['tactical']['total_detections']} | Saved: {report['tactical']['saved_intelligence']} | Rate: {report['tactical']['detection_rate']:.1%}")
        self.logger.info(f"Final Threat: {report['tactical']['final_threat_level']} | Report: {rf}")
        self.logger.info("="*80)
