#!/usr/bin/env python3
"""
================================================================================
PROFESSIONAL DRONE DETECTION SYSTEM v2.1 - REAL-TIME VERSION
================================================================================
Copyright (c) 2025 TEERATHAP INDUSTRY Co., Ltd.
All Rights Reserved.

Real-time Version - Processing Every Frame
================================================================================
"""

import cv2
import numpy as np
import logging
import argparse
import time
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from ultralytics import YOLO
import threading
from collections import deque


# ==================== SOFTWARE INFORMATION ====================
SOFTWARE_INFO = {
    "name": "Professional Drone Detection System - Real-Time",
    "version": "2.1.0-RT",
    "company": "TEERATHAP INDUSTRY Co., Ltd.",
    "copyright": "Copyright (c) 2025 TEERATHAP INDUSTRY. All Rights Reserved.",
    "license": "Proprietary Software License",
    "contact": "contact@teerathapindustry.com",
    "website": "https://www.teerathapindustry.com"
}


# ==================== Real-time Configuration Classes ====================
@dataclass
class DetectionConfig:
    """Real-time configuration for every frame processing"""
    model_path: str = 'best.pt'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    target_classes: List[str] = None
    auto_save_detections: bool = False
    save_directory: str = "detections"
    # Real-time optimizations
    detection_interval: int = 1  # Process EVERY frame
    use_threading: bool = True
    max_queue_size: int = 2
    use_gpu_optimization: bool = True
    half_precision: bool = True
    
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = ['drone']


@dataclass
class DisplayConfig:
    """Display configuration optimized for real-time"""
    window_name: str = 'TEERATHAP INDUSTRY - Real-Time Drone Detection v2.1'
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False
    
    # Colors
    primary_color: Tuple[int, int, int] = (0, 255, 0)
    secondary_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (0, 165, 255)
    alert_color: Tuple[int, int, int] = (0, 0, 255)
    
    box_thickness: int = 2
    font_scale: float = 0.6
    simplified_ui: bool = True


# ==================== Real-time Logging System ====================
class RealTimeLogger:
    """Lightweight logging for real-time performance"""
    
    def __init__(self, log_level: str = 'WARNING'):
        self.logger = logging.getLogger('TEERATHAP_REALTIME_DETECTION')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger


# ==================== Real-time Detection System ====================
class RealTimeDroneDetectionSystem:
    """
    Real-time Drone Detection System
    Processing every frame for maximum responsiveness
    """
    
    def __init__(self, detection_config: DetectionConfig, display_config: DisplayConfig, log_level: str = 'WARNING'):
        self.detection_config = detection_config
        self.display_config = display_config
        
        # Initialize logging
        self.logger_system = RealTimeLogger(log_level)
        self.logger = self.logger_system.get_logger()
        
        # System components
        self.model = None
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_detections = []
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Threading for real-time processing
        if self.detection_config.use_threading:
            self.detection_queue = deque(maxlen=detection_config.max_queue_size)
            self.result_queue = deque(maxlen=detection_config.max_queue_size)
            self.detection_thread = None
            self.thread_running = False
        
        # Statistics
        self.session_stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'detection_count': 0
        }
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.detection_config.save_directory, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def initialize_model(self) -> bool:
        """Initialize YOLO model with real-time optimizations"""
        try:
            model_path = Path(self.detection_config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.logger.info("Loading real-time optimized YOLO model...")
            self.model = YOLO(str(model_path))
            
            # Real-time optimizations
            self.model.overrides['verbose'] = False
            if self.detection_config.use_gpu_optimization:
                self.model.overrides['device'] = 'cuda'
            
            # Enable half precision for speed
            if self.detection_config.half_precision:
                try:
                    self.model.model.half()
                    self.logger.info("Half precision (FP16) enabled")
                except:
                    self.logger.warning("Half precision not supported")
            
            # Warm up model
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
            
            self.logger.info("Real-time model ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera for maximum performance"""
        try:
            self.logger.info(f"Initializing real-time camera {camera_id}...")
            
            # Try different backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(camera_id, backend)
                if self.cap.isOpened():
                    break
            
            if not self.cap.isOpened():
                raise IOError("Cannot access camera")
            
            # Real-time camera settings
            settings = [
                (cv2.CAP_PROP_FRAME_WIDTH, 960),
                (cv2.CAP_PROP_FRAME_HEIGHT, 540),
                (cv2.CAP_PROP_FPS, 120),
                (cv2.CAP_PROP_BUFFERSIZE, 1),
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')),
                (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
                (cv2.CAP_PROP_AUTOFOCUS, 0),
            ]
            
            for prop, value in settings:
                try:
                    self.cap.set(prop, value)
                except:
                    pass
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                raise IOError("Camera opened but cannot capture frames")
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Real-time camera ready: {width}x{height} @ {fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def detection_worker(self):
        """Real-time detection worker thread"""
        while self.thread_running:
            if len(self.detection_queue) > 0:
                frame = self.detection_queue.popleft()
                
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, 
                                   conf=self.detection_config.confidence_threshold,
                                   iou=self.detection_config.iou_threshold,
                                   max_det=self.detection_config.max_detections,
                                   verbose=False,
                                   stream=False,
                                   vid_stride=1)
                
                # Process results
                detections = self.process_detections_fast(results)
                
                detection_time = time.time() - start_time
                result_data = {
                    'detections': detections,
                    'processing_time': detection_time
                }
                
                # Keep only latest result
                self.result_queue.clear()
                self.result_queue.append(result_data)
            else:
                time.sleep(0.0001)
    
    def process_detections_fast(self, results) -> List[Dict]:
        """Ultra-fast detection processing"""
        detections = []
        
        if not results:
            return detections
        
        result = results[0]
        if result.boxes is None:
            return detections
        
        # Vectorized processing
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by confidence
        conf_mask = confidences >= self.detection_config.confidence_threshold
        valid_boxes = boxes[conf_mask]
        valid_confs = confidences[conf_mask]
        valid_classes = classes[conf_mask]
        
        # Target class set for fast lookup
        target_set = {cls.lower() for cls in self.detection_config.target_classes}
        
        for i in range(len(valid_boxes)):
            cls_id = valid_classes[i]
            class_name = self.model.names[cls_id]
            
            if class_name.lower() in target_set:
                x1, y1, x2, y2 = valid_boxes[i].astype(int)
                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(valid_confs[i]),
                    'class_name': class_name
                }
                detections.append(detection)
        
        return detections
    
    def draw_real_time_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float, detection_time: float = 0) -> np.ndarray:
        """Real-time overlay drawing"""
        if self.display_config.simplified_ui:
            return self.draw_minimal_overlay(frame, detections, fps, detection_time)
        else:
            return self.draw_standard_overlay(frame, detections, fps, detection_time)
    
    def draw_minimal_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float, detection_time: float) -> np.ndarray:
        """Minimal overlay for maximum speed"""
        height, width = frame.shape[:2]
        
        # Simple header
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        
        status_text = f"REAL-TIME | FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms | Targets: {len(detections)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Simple detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{detection['confidence']:.0%}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def draw_standard_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float, detection_time: float) -> np.ndarray:
        """Standard overlay for real-time processing"""
        height, width = frame.shape[:2]
        
        # Header
        header_height = 70
        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        
        cv2.putText(frame, "TEERATHAP INDUSTRY - REAL-TIME DETECTION", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_line = f"FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms | Targets: {len(detections)}"
        cv2.putText(frame, status_line, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection boxes with corner markers
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Main box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"#{i+1} {detection['class_name']} {conf:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(frame, (x1, y1-20), (x1 + label_size[0] + 8, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Corner markers
            corner_size = 15
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), (255, 0, 0), 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), (255, 0, 0), 3)
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), (255, 0, 0), 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), (255, 0, 0), 3)
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), (255, 0, 0), 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), (255, 0, 0), 3)
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), (255, 0, 0), 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), (255, 0, 0), 3)
        
        # Stats in corner
        runtime = time.time() - self.session_stats['start_time']
        stats = [
            f"Runtime: {runtime:.0f}s",
            f"Frames: {self.session_stats['total_frames']:,}",
            f"Detections: {self.session_stats['detection_count']:,}",
            f"Speed: {detection_time*1000:.1f}ms"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (width-200, header_height + 25 + i*18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def calculate_fps(self) -> float:
        """Calculate real-time FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return (len(self.fps_counter) - 1) / time_diff
        
        return 0.0
    
    def run_detection_system(self) -> None:
        """Run real-time detection system - processing every frame"""
        # Initialize system
        if not self.initialize_model():
            return
        
        if not self.initialize_camera():
            return
        
        # Setup window
        cv2.namedWindow(self.display_config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_config.window_name, 
                        self.display_config.window_width, 
                        self.display_config.window_height)
        
        # Start detection thread
        if self.detection_config.use_threading:
            self.thread_running = True
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()
        
        self.logger.info("🚀 REAL-TIME DRONE DETECTION STARTED - PROCESSING EVERY FRAME")
        self.logger.info("Controls: [Q]uit | [S]implify UI | [F]ullscreen")
        
        detection_times = deque(maxlen=30)
        
        try:
            while True:
                frame_start = time.time()
                
                # Capture frame
                success, frame = self.cap.read()
                if not success:
                    continue
                
                self.frame_count += 1
                self.session_stats['total_frames'] += 1
                
                # Real-time detection - every frame
                if self.detection_config.use_threading:
                    # Send frame for processing
                    self.detection_queue.clear()
                    self.detection_queue.append(frame.copy())
                    
                    # Get results
                    if len(self.result_queue) > 0:
                        result_data = self.result_queue.popleft()
                        self.last_detections = result_data['detections']
                        detection_times.append(result_data['processing_time'])
                        self.session_stats['detection_count'] += len(self.last_detections)
                    
                    detections = self.last_detections
                else:
                    # Direct processing
                    detection_start = time.time()
                    results = self.model(frame, 
                                       conf=self.detection_config.confidence_threshold,
                                       iou=self.detection_config.iou_threshold,
                                       max_det=self.detection_config.max_detections,
                                       verbose=False)
                    detections = self.process_detections_fast(results)
                    detection_time = time.time() - detection_start
                    detection_times.append(detection_time)
                    
                    self.last_detections = detections
                    self.session_stats['detection_count'] += len(detections)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Average detection time
                avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                
                # Draw overlay
                frame = self.draw_real_time_overlay(frame, detections, fps, avg_detection_time)
                
                # Display
                cv2.imshow(self.display_config.window_name, frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Toggle UI
                    self.display_config.simplified_ui = not self.display_config.simplified_ui
                    self.logger.info(f"Simplified UI: {self.display_config.simplified_ui}")
                elif key == ord('f'):  # Fullscreen
                    self.display_config.fullscreen = not self.display_config.fullscreen
                    if self.display_config.fullscreen:
                        cv2.setWindowProperty(self.display_config.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.display_config.window_name, 
                                            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up system resources"""
        # Stop detection thread
        if self.detection_config.use_threading and self.thread_running:
            self.thread_running = False
            if self.detection_thread:
                self.detection_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        runtime = time.time() - self.session_stats['start_time']
        avg_fps = self.session_stats['total_frames'] / runtime if runtime > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("REAL-TIME DETECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Runtime: {runtime:.1f}s")
        self.logger.info(f"Frames Processed: {self.session_stats['total_frames']:,}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Total Detections: {self.session_stats['detection_count']}")
        self.logger.info("=" * 60)


# ==================== Command Line Interface ====================
def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(description="Real-Time Drone Detection System")
    
    parser.add_argument('--model', '-m', default='best.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--classes', nargs='+', default=['drone'], help='Target classes')
    parser.add_argument('--no-threading', action='store_true', help='Disable threading')
    parser.add_argument('--simplified', action='store_true', help='Use minimal UI for maximum speed')
    parser.add_argument('--half-precision', action='store_true', help='Enable FP16 for speed boost')
    parser.add_argument('--gpu', action='store_true', help='Force GPU acceleration')
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    return parser


# ==================== Main Entry Point ====================
def main():
    """Main application entry point"""
    print("=" * 60)
    print("REAL-TIME DRONE DETECTION SYSTEM v2.1-RT")
    print("TEERATHAP INDUSTRY Co., Ltd.")
    print("Processing Every Frame for Maximum Responsiveness")
    print("=" * 60)
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configurations
    detection_config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        target_classes=args.classes,
        detection_interval=1,  # Always process every frame
        use_threading=not args.no_threading,
        auto_save_detections=False,
        use_gpu_optimization=args.gpu,
        half_precision=args.half_precision
    )
    
    display_config = DisplayConfig(
        simplified_ui=args.simplified
    )
    
    # Display configuration
    print(f"🚀 REAL-TIME Configuration:")
    print(f"   Processing: Every Frame (Real-time)")
    print(f"   Threading: {'Enabled' if not args.no_threading else 'Disabled'}")
    print(f"   GPU: {'Enabled' if args.gpu else 'Auto-detect'}")
    print(f"   Half Precision: {'Enabled' if args.half_precision else 'Disabled'}")
    print(f"   UI Mode: {'Minimal' if args.simplified else 'Standard'}")
    print("=" * 60)
    
    # Run system
    system = RealTimeDroneDetectionSystem(detection_config, display_config, args.log_level)
    system.run_detection_system()


if __name__ == "__main__":
    main()
