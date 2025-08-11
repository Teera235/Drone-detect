#!/usr/bin/env python3
"""
Professional Military-Grade Drone Detection System
===================================================
Developed for tactical surveillance and threat assessment operations.
Features advanced detection algorithms, multi-source input, and comprehensive logging.

Author: Advanced Defense Systems
Version: 2.0
Classification: RESTRICTED
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
import json
import threading
from collections import deque, defaultdict
from datetime import datetime, timezone
import os
import argparse
from pathlib import Path
import signal
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import socket
import sqlite3
from contextlib import contextmanager

# ===============================
# SYSTEM CONFIGURATION
# ===============================

@dataclass
class SystemConfig:
    """System configuration parameters"""
    model_path: str = "best.pt"
    camera_sources: List[str] = None
    confidence_threshold: float = 0.6
    target_classes: List[str] = None
    log_level: str = "INFO"
    output_dir: str = "./detection_logs"
    database_path: str = "./detections.db"
    alert_threshold: int = 3  # Number of consecutive detections to trigger alert
    frame_skip: int = 1  # Process every nth frame for performance
    max_tracking_age: int = 30  # Frames to keep tracking lost objects
    detection_zone: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    enable_recording: bool = False
    recording_duration: int = 30  # seconds
    alert_webhook_url: Optional[str] = None
    network_broadcast: bool = False
    broadcast_port: int = 5555

    def __post_init__(self):
        if self.camera_sources is None:
            self.camera_sources = ["rtsp://192.168.1.50:8080/h264_pcm.sdp"]
        if self.target_classes is None:
            self.target_classes = ["drone", "uav", "quadcopter", "aircraft"]

@dataclass
class Detection:
    """Detection data structure"""
    timestamp: float
    camera_id: str
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: int
    tracking_id: Optional[int] = None

class ThreatLevel:
    """Threat assessment levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MilitaryDroneDetector:
    """Professional military-grade drone detection system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        self.models = {}
        self.captures = {}
        self.detection_history = defaultdict(deque)
        self.tracking_objects = {}
        self.next_track_id = 1
        self.alert_states = defaultdict(int)
        self.recording_threads = {}
        
        # Setup logging
        self._setup_logging()
        
        # Setup database
        self._setup_database()
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Military Drone Detection System initialized")
        
    def _setup_logging(self):
        """Configure professional logging system"""
        log_format = '[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for persistent logging
        log_file = Path(self.config.output_dir) / f"detection_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        self.logger = logging.getLogger("DroneDetector")
        self.logger.addHandler(file_handler)
        
    def _setup_database(self):
        """Initialize SQLite database for detection logging"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    camera_id TEXT,
                    class_name TEXT,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    center_x INTEGER,
                    center_y INTEGER,
                    area INTEGER,
                    tracking_id INTEGER,
                    threat_level TEXT,
                    alert_triggered BOOLEAN
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.config.database_path, timeout=10)
        try:
            yield conn
        finally:
            conn.close()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        sys.exit(0)
    
    def load_models(self):
        """Load YOLO models for all camera sources"""
        self.logger.info("Loading YOLO models...")
        try:
            # Load primary model
            model = YOLO(self.config.model_path)
            
            # Assign model to all cameras
            for i, source in enumerate(self.config.camera_sources):
                camera_id = f"CAM_{i:02d}"
                self.models[camera_id] = model
                self.logger.info(f"Model loaded for {camera_id}: {source}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def initialize_cameras(self):
        """Initialize all camera sources"""
        self.logger.info("Initializing camera sources...")
        
        for i, source in enumerate(self.config.camera_sources):
            camera_id = f"CAM_{i:02d}"
            
            # Convert string numbers to int for local cameras
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}: {source}")
                continue
                
            # Set optimal camera properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to set high resolution if possible
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            self.captures[camera_id] = cap
            self.logger.info(f"Camera {camera_id} initialized: {source}")
    
    def _calculate_threat_level(self, detections: List[Detection], camera_id: str) -> str:
        """Calculate threat level based on detection characteristics"""
        if not detections:
            return ThreatLevel.LOW
            
        # Factors for threat assessment
        max_confidence = max(d.confidence for d in detections)
        detection_count = len(detections)
        consecutive_detections = self.alert_states[camera_id]
        
        # Large objects (closer drones) are more threatening
        max_area = max(d.area for d in detections) if detections else 0
        
        # Calculate threat score
        threat_score = 0
        
        if max_confidence > 0.9:
            threat_score += 3
        elif max_confidence > 0.7:
            threat_score += 2
        else:
            threat_score += 1
            
        if detection_count > 3:
            threat_score += 2
        elif detection_count > 1:
            threat_score += 1
            
        if consecutive_detections > 10:
            threat_score += 3
        elif consecutive_detections > 5:
            threat_score += 2
            
        if max_area > 50000:  # Large drone
            threat_score += 2
        elif max_area > 20000:  # Medium drone
            threat_score += 1
            
        # Determine threat level
        if threat_score >= 8:
            return ThreatLevel.CRITICAL
        elif threat_score >= 6:
            return ThreatLevel.HIGH
        elif threat_score >= 3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _simple_tracking(self, current_detections: List[Detection], camera_id: str) -> List[Detection]:
        """Simple object tracking based on proximity"""
        if camera_id not in self.tracking_objects:
            self.tracking_objects[camera_id] = {}
            
        tracked_objects = self.tracking_objects[camera_id]
        matched_detections = []
        
        # Match current detections with tracked objects
        for detection in current_detections:
            best_match = None
            best_distance = float('inf')
            
            for track_id, (last_center, age) in tracked_objects.items():
                distance = np.sqrt((detection.center[0] - last_center[0])**2 + 
                                 (detection.center[1] - last_center[1])**2)
                
                if distance < best_distance and distance < 100:  # 100 pixel threshold
                    best_distance = distance
                    best_match = track_id
            
            if best_match:
                detection.tracking_id = best_match
                tracked_objects[best_match] = (detection.center, 0)
            else:
                # New object
                detection.tracking_id = self.next_track_id
                tracked_objects[self.next_track_id] = (detection.center, 0)
                self.next_track_id += 1
                
            matched_detections.append(detection)
        
        # Age unmatched tracks
        for track_id in list(tracked_objects.keys()):
            center, age = tracked_objects[track_id]
            if not any(d.tracking_id == track_id for d in matched_detections):
                age += 1
                if age > self.config.max_tracking_age:
                    del tracked_objects[track_id]
                else:
                    tracked_objects[track_id] = (center, age)
        
        return matched_detections
    
    def _log_detection(self, detection: Detection, threat_level: str, alert_triggered: bool):
        """Log detection to database"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO detections (
                        timestamp, camera_id, class_name, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        center_x, center_y, area, tracking_id,
                        threat_level, alert_triggered
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection.timestamp, detection.camera_id, detection.class_name,
                    detection.confidence, *detection.bbox, *detection.center,
                    detection.area, detection.tracking_id, threat_level, alert_triggered
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database logging error: {e}")
    
    def _trigger_alert(self, detections: List[Detection], camera_id: str, threat_level: str):
        """Trigger alert systems"""
        alert_msg = (f"🚨 DRONE ALERT - {threat_level} THREAT 🚨\n"
                    f"Camera: {camera_id}\n"
                    f"Detections: {len(detections)}\n"
                    f"Time: {datetime.now().isoformat()}\n"
                    f"Max Confidence: {max(d.confidence for d in detections):.1%}")
        
        self.logger.warning(alert_msg.replace('\n', ' | '))
        
        # Save alert to file
        alert_file = Path(self.config.output_dir) / "alerts.log"
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {alert_msg}\n\n")
        
        # Trigger recording if enabled
        if self.config.enable_recording:
            self._start_recording(camera_id)
    
    def _start_recording(self, camera_id: str):
        """Start recording for specified duration"""
        if camera_id in self.recording_threads:
            return  # Already recording
            
        def record():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(self.config.output_dir) / f"alert_recording_{camera_id}_{timestamp}.mp4"
            
            cap = self.captures[camera_id]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Get frame dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            
            start_time = time.time()
            while time.time() - start_time < self.config.recording_duration:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                time.sleep(1/fps)
            
            out.release()
            self.logger.info(f"Recording saved: {output_file}")
            del self.recording_threads[camera_id]
        
        self.recording_threads[camera_id] = threading.Thread(target=record)
        self.recording_threads[camera_id].start()
    
    def process_frame(self, frame: np.ndarray, camera_id: str, frame_count: int) -> Tuple[np.ndarray, List[Detection]]:
        """Process single frame for detections"""
        if frame_count % self.config.frame_skip != 0:
            return frame, []
            
        # Apply detection zone if configured
        detection_frame = frame
        zone_offset = (0, 0)
        
        if self.config.detection_zone:
            x1, y1, x2, y2 = self.config.detection_zone
            detection_frame = frame[y1:y2, x1:x2]
            zone_offset = (x1, y1)
            
            # Draw detection zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "DETECTION ZONE", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Run YOLO detection
        model = self.models[camera_id]
        results = model(detection_frame, conf=self.config.confidence_threshold, verbose=False)
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                class_name = model.names[clss[i]].lower()
                if any(target.lower() in class_name for target in self.config.target_classes):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    
                    # Adjust coordinates for detection zone
                    x1 += zone_offset[0]
                    y1 += zone_offset[1]
                    x2 += zone_offset[0]
                    y2 += zone_offset[1]
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = Detection(
                        timestamp=time.time(),
                        camera_id=camera_id,
                        class_name=class_name,
                        confidence=confs[i],
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        area=area
                    )
                    detections.append(detection)
        
        # Apply tracking
        detections = self._simple_tracking(detections, camera_id)
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(detections, camera_id)
        
        # Update alert state
        if detections:
            self.alert_states[camera_id] += 1
        else:
            self.alert_states[camera_id] = max(0, self.alert_states[camera_id] - 1)
        
        # Trigger alerts if necessary
        alert_triggered = False
        if (len(detections) > 0 and 
            self.alert_states[camera_id] >= self.config.alert_threshold and
            threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]):
            
            self._trigger_alert(detections, camera_id, threat_level)
            alert_triggered = True
        
        # Log detections
        for detection in detections:
            self._log_detection(detection, threat_level, alert_triggered)
        
        # Store detection history
        if camera_id not in self.detection_history:
            self.detection_history[camera_id] = deque(maxlen=100)
        self.detection_history[camera_id].append(len(detections))
        
        return frame, detections, threat_level
    
    def draw_interface(self, frame: np.ndarray, detections: List[Detection], 
                      camera_id: str, fps: float, threat_level: str) -> np.ndarray:
        """Draw professional military interface"""
        height, width = frame.shape[:2]
        
        # Threat level colors
        threat_colors = {
            ThreatLevel.LOW: (0, 255, 0),
            ThreatLevel.MEDIUM: (0, 255, 255),
            ThreatLevel.HIGH: (0, 165, 255),
            ThreatLevel.CRITICAL: (0, 0, 255)
        }
        
        threat_color = threat_colors.get(threat_level, (255, 255, 255))
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), threat_color, 3)
            
            # Crosshair
            center_x, center_y = detection.center
            crosshair_size = 20
            cv2.line(frame, (center_x - crosshair_size, center_y),
                    (center_x + crosshair_size, center_y), threat_color, 2)
            cv2.line(frame, (center_x, center_y - crosshair_size),
                    (center_x, center_y + crosshair_size), threat_color, 2)
            
            # Detection info
            info_text = f"ID:{detection.tracking_id} {detection.class_name.upper()} {detection.confidence:.0%}"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Info background
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), threat_color, -1)
            cv2.putText(frame, info_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Status panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # System status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        status_lines = [
            f"🔴 TACTICAL DRONE DETECTION SYSTEM - {camera_id}",
            f"⏰ {timestamp} | 📊 FPS: {fps:.1f}",
            f"🎯 TARGETS: {len(detections)} | ⚠️  THREAT: {threat_level}",
            f"📈 CONSECUTIVE ALERTS: {self.alert_states[camera_id]}"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = 25 + (i * 25)
            cv2.putText(frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Threat level indicator
        threat_indicator_size = 30
        cv2.rectangle(frame, (width - 200, 10), 
                     (width - 200 + threat_indicator_size, 10 + threat_indicator_size),
                     threat_color, -1)
        cv2.putText(frame, threat_level, (width - 190, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2)
        
        return frame
    
    def run_detection(self):
        """Main detection loop"""
        self.logger.info("Starting detection system...")
        self.running = True
        
        fps_counters = {camera_id: deque(maxlen=30) for camera_id in self.captures.keys()}
        frame_counts = defaultdict(int)
        
        try:
            while self.running:
                for camera_id, cap in self.captures.items():
                    start_time = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(f"Failed to read from {camera_id}")
                        continue
                    
                    frame_counts[camera_id] += 1
                    
                    # Process frame
                    try:
                        processed_frame, detections, threat_level = self.process_frame(
                            frame, camera_id, frame_counts[camera_id]
                        )
                    except Exception as e:
                        self.logger.error(f"Processing error for {camera_id}: {e}")
                        processed_frame, detections, threat_level = frame, [], ThreatLevel.LOW
                    
                    # Calculate FPS
                    fps_counters[camera_id].append(time.time())
                    if len(fps_counters[camera_id]) > 1:
                        fps = (len(fps_counters[camera_id]) - 1) / \
                              (fps_counters[camera_id][-1] - fps_counters[camera_id][0])
                    else:
                        fps = 0.0
                    
                    # Draw interface
                    display_frame = self.draw_interface(
                        processed_frame, detections, camera_id, fps, threat_level
                    )
                    
                    # Display
                    window_name = f"🎯 MILITARY DRONE DETECTION - {camera_id}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle key input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # ESC
                        self.running = False
                        break
                    elif key == ord('s'):  # Screenshot
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        screenshot_path = Path(self.config.output_dir) / f"screenshot_{camera_id}_{timestamp}.jpg"
                        cv2.imwrite(str(screenshot_path), display_frame)
                        self.logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('r'):  # Manual recording
                        self._start_recording(camera_id)
                        self.logger.info(f"Manual recording started for {camera_id}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Critical error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up system resources...")
        
        # Release cameras
        for camera_id, cap in self.captures.items():
            cap.release()
            self.logger.info(f"Released camera {camera_id}")
        
        # Wait for recording threads
        for camera_id, thread in self.recording_threads.items():
            thread.join(timeout=5)
            self.logger.info(f"Recording thread {camera_id} finished")
        
        cv2.destroyAllWindows()
        self.logger.info("System shutdown complete")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Military-Grade Drone Detection System")
    parser.add_argument("--model", default="best.pt", help="Path to YOLO model")
    parser.add_argument("--cameras", nargs="+", default=["0"], help="Camera sources")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--classes", nargs="+", default=["drone"], help="Target classes")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output", default="./detection_logs", help="Output directory")
    parser.add_argument("--enable-recording", action="store_true", help="Enable alert recording")
    parser.add_argument("--detection-zone", nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                       help="Detection zone coordinates")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig(
        model_path=args.model,
        camera_sources=args.cameras,
        confidence_threshold=args.conf,
        target_classes=args.classes,
        log_level=args.log_level.upper(),
        output_dir=args.output,
        enable_recording=args.enable_recording,
        detection_zone=tuple(args.detection_zone) if args.detection_zone else None
    )
    
    # Initialize and run detection system
    detector = MilitaryDroneDetector(config)
    
    try:
        detector.load_models()
        detector.initialize_cameras()
        
        if not detector.captures:
            print("❌ No cameras available. Exiting.")
            return
            
        print("🎯 MILITARY DRONE DETECTION SYSTEM ACTIVE")
        print("📋 Controls:")
        print("   Q/ESC - Exit system")
        print("   S     - Take screenshot")
        print("   R     - Start manual recording")
        print("=" * 50)
        
        detector.run_detection()
        
    except Exception as e:
        detector.logger.error(f"System failure: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
