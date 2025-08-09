#!/usr/bin/env python3
"""
================================================================================
PROFESSIONAL DRONE DETECTION SYSTEM v2.1 - OPTIMIZED VERSION
================================================================================
Copyright (c) 2025 TEERATHAP INDUSTRY Co., Ltd.
All Rights Reserved.

High-Performance Real-time Version - Optimized for Speed
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
    "name": "Professional Drone Detection System - Optimized",
    "version": "2.1.0-OPT",
    "company": "TEERATHAP INDUSTRY Co., Ltd.",
    "copyright": "Copyright (c) 2025 TEERATHAP INDUSTRY. All Rights Reserved.",
    "license": "Proprietary Software License",
    "contact": "contact@teerathapindustry.com",
    "website": "https://www.teerathapindustry.com"
}


# ==================== Optimized Configuration Classes ====================
@dataclass
class DetectionConfig:
    """Optimized configuration for high-speed detection"""
    model_path: str = 'best.pt'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100  # Reduced for speed
    target_classes: List[str] = None
    auto_save_detections: bool = False  # Disabled for speed
    save_directory: str = "detections"
    # Performance optimizations
    detection_interval: int = 2  # Process every N frames
    use_threading: bool = True
    max_queue_size: int = 5
    
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = ['drone']


@dataclass
class DisplayConfig:
    """Optimized display configuration for performance"""
    window_name: str = 'TEERATHAP INDUSTRY - High-Speed Drone Detection v2.1'
    # Optimized resolution for performance
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False
    
    # Simplified color scheme for faster rendering
    primary_color: Tuple[int, int, int] = (0, 255, 0)      # Green
    secondary_color: Tuple[int, int, int] = (255, 255, 255) # White
    accent_color: Tuple[int, int, int] = (0, 165, 255)     # Orange
    alert_color: Tuple[int, int, int] = (0, 0, 255)        # Red
    
    box_thickness: int = 2  # Reduced thickness
    font_scale: float = 0.6  # Smaller font
    simplified_ui: bool = True  # Enable simplified UI


# ==================== Optimized Logging System ====================
class OptimizedLogger:
    """Lightweight logging system for high performance"""
    
    def __init__(self, log_level: str = 'WARNING'):  # Reduced default logging
        self.logger = logging.getLogger('TEERATHAP_DRONE_DETECTION_OPT')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Minimal logging setup for performance
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger


# ==================== High-Performance Detection System ====================
class OptimizedDroneDetectionSystem:
    """
    High-Performance Drone Detection System
    Optimized for real-time processing with minimal latency
    """
    
    def __init__(self, detection_config: DetectionConfig, display_config: DisplayConfig, log_level: str = 'WARNING'):
        self.detection_config = detection_config
        self.display_config = display_config
        
        # Initialize optimized logging
        self.logger_system = OptimizedLogger(log_level)
        self.logger = self.logger_system.get_logger()
        
        # System components
        self.model = None
        self.cap = None
        
        # Performance optimization variables
        self.frame_count = 0
        self.last_detection_frame = None
        self.last_detections = []
        self.fps_counter = deque(maxlen=30)  # Rolling average
        self.last_fps_time = time.time()
        
        # Threading for detection
        if self.detection_config.use_threading:
            self.detection_queue = deque(maxlen=detection_config.max_queue_size)
            self.result_queue = deque(maxlen=detection_config.max_queue_size)
            self.detection_thread = None
            self.thread_running = False
        
        # Simplified statistics
        self.session_stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'detection_count': 0
        }
        
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create necessary directories efficiently"""
        os.makedirs(self.detection_config.save_directory, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def initialize_model(self) -> bool:
        """Initialize YOLO model with optimization"""
        try:
            model_path = Path(self.detection_config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.logger.info("Loading optimized YOLO model...")
            self.model = YOLO(str(model_path))
            
            # Optimize model for inference speed
            self.model.overrides['verbose'] = False
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera with performance optimizations"""
        try:
            self.logger.info(f"Initializing camera {camera_id}...")
            
            # Use fastest backend for Windows
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                raise IOError("Cannot access camera")
            
            # Optimized camera settings for speed
            camera_settings = [
                (cv2.CAP_PROP_FRAME_WIDTH, 1280),   # Lower resolution for speed
                (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                (cv2.CAP_PROP_FPS, 60),             # Higher FPS
                (cv2.CAP_PROP_BUFFERSIZE, 1),       # Minimize buffer for low latency
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG for speed
            ]
            
            for prop, value in camera_settings:
                self.cap.set(prop, value)
            
            # Verify camera is working
            ret, test_frame = self.cap.read()
            if not ret:
                raise IOError("Camera opened but cannot capture frames")
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def detection_worker(self):
        """Background thread for running detections"""
        while self.thread_running:
            if len(self.detection_queue) > 0:
                frame = self.detection_queue.popleft()
                
                # Run detection
                results = self.model(frame, 
                                   conf=self.detection_config.confidence_threshold,
                                   iou=self.detection_config.iou_threshold,
                                   max_det=self.detection_config.max_detections,
                                   verbose=False)
                
                # Process results
                detections = self.process_detections(results)
                
                # Store results
                if len(self.result_queue) < self.detection_config.max_queue_size:
                    self.result_queue.append(detections)
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def process_detections(self, results) -> List[Dict]:
        """Optimized detection processing"""
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            # Vectorized processing for speed
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # Filter by confidence in vectorized manner
            valid_indices = confidences >= self.detection_config.confidence_threshold
            
            for i in np.where(valid_indices)[0]:
                cls_id = classes[i]
                class_name = self.model.names[cls_id]
                
                # Quick class check
                if class_name.lower() in [cls.lower() for cls in self.detection_config.target_classes]:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidences[i]),
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_optimized_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        """Highly optimized overlay drawing for maximum performance"""
        if self.display_config.simplified_ui:
            return self.draw_minimal_overlay(frame, detections, fps)
        else:
            return self.draw_standard_overlay(frame, detections, fps)
    
    def draw_minimal_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        """Minimal overlay for maximum performance"""
        height, width = frame.shape[:2]
        
        # Simple header
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"TEERATHAP INDUSTRY - DRONE DETECTION | FPS: {fps:.1f}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)} | Status: {'ACTIVE' if detections else 'SCANNING'}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Simple detection boxes
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Simple green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Simple label
            label = f"{detection['class_name']} {conf:.1%}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        return frame
    
    def draw_standard_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        """Standard overlay with reduced complexity"""
        height, width = frame.shape[:2]
        
        # Simplified header
        header_height = 80
        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "TEERATHAP INDUSTRY - HIGH-SPEED DETECTION", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {len(detections)}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with background
            label = f"{detection['class_name']} {conf:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Simple stats in corner
        stats_text = [
            f"Runtime: {time.time() - self.session_stats['start_time']:.0f}s",
            f"Frames: {self.session_stats['total_frames']}",
            f"Total Detections: {self.session_stats['detection_count']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (width-250, header_height + 25 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def calculate_fps(self) -> float:
        """Optimized FPS calculation"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return (len(self.fps_counter) - 1) / time_diff
        
        return 0.0
    
    def run_detection_system(self) -> None:
        """Run the high-performance detection system"""
        # System initialization
        if not self.initialize_model():
            return
        
        if not self.initialize_camera():
            return
        
        # Setup display window
        cv2.namedWindow(self.display_config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_config.window_name, 
                        self.display_config.window_width, 
                        self.display_config.window_height)
        
        # Start detection thread if enabled
        if self.detection_config.use_threading:
            self.thread_running = True
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()
        
        self.logger.info("HIGH-SPEED DRONE DETECTION SYSTEM STARTED")
        self.logger.info("Controls: [Q]uit | [S]implify UI | [F]ullscreen")
        
        try:
            while True:
                # Capture frame
                success, frame = self.cap.read()
                if not success:
                    continue
                
                self.frame_count += 1
                self.session_stats['total_frames'] += 1
                
                # Detection logic with frame skipping
                if self.detection_config.use_threading:
                    # Threading mode
                    if self.frame_count % self.detection_config.detection_interval == 0:
                        if len(self.detection_queue) < self.detection_config.max_queue_size:
                            self.detection_queue.append(frame.copy())
                    
                    # Get latest results
                    if len(self.result_queue) > 0:
                        self.last_detections = self.result_queue.popleft()
                        self.session_stats['detection_count'] += len(self.last_detections)
                    
                    detections = self.last_detections
                else:
                    # Non-threading mode with frame skipping
                    if self.frame_count % self.detection_config.detection_interval == 0:
                        results = self.model(frame, 
                                           conf=self.detection_config.confidence_threshold,
                                           iou=self.detection_config.iou_threshold,
                                           max_det=self.detection_config.max_detections,
                                           verbose=False)
                        detections = self.process_detections(results)
                        self.last_detections = detections
                        self.session_stats['detection_count'] += len(detections)
                    else:
                        detections = self.last_detections
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw overlay (optimized)
                frame = self.draw_optimized_overlay(frame, detections, fps)
                
                # Display frame
                cv2.imshow(self.display_config.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Toggle simplified UI
                    self.display_config.simplified_ui = not self.display_config.simplified_ui
                    self.logger.info(f"Simplified UI: {self.display_config.simplified_ui}")
                elif key == ord('f'):  # Toggle fullscreen
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
        """Optimized system cleanup"""
        # Stop detection thread
        if self.detection_config.use_threading and self.thread_running:
            self.thread_running = False
            if self.detection_thread:
                self.detection_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Log final statistics
        runtime = time.time() - self.session_stats['start_time']
        avg_fps = self.session_stats['total_frames'] / runtime if runtime > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("HIGH-SPEED DETECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Runtime: {runtime:.1f}s")
        self.logger.info(f"Frames Processed: {self.session_stats['total_frames']:,}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Total Detections: {self.session_stats['detection_count']}")
        self.logger.info("=" * 60)


# ==================== Optimized CLI ====================
def create_optimized_parser():
    """Create optimized command line parser"""
    parser = argparse.ArgumentParser(description="High-Performance Drone Detection System")
    
    parser.add_argument('--model', '-m', default='best.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--classes', nargs='+', default=['drone'], help='Target classes')
    parser.add_argument('--detection-interval', type=int, default=2, help='Process every N frames (higher = faster)')
    parser.add_argument('--no-threading', action='store_true', help='Disable threading')
    parser.add_argument('--simplified', action='store_true', help='Use minimal UI')
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    return parser


# ==================== Main Entry Point ====================
def main():
    """Optimized main function"""
    print("=" * 60)
    print("HIGH-SPEED DRONE DETECTION SYSTEM v2.1-OPT")
    print("TEERATHAP INDUSTRY Co., Ltd.")
    print("Optimized for Real-time Performance")
    print("=" * 60)
    
    parser = create_optimized_parser()
    args = parser.parse_args()
    
    # Create optimized configurations
    detection_config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        target_classes=args.classes,
        detection_interval=args.detection_interval,
        use_threading=not args.no_threading,
        auto_save_detections=False  # Disabled for performance
    )
    
    display_config = DisplayConfig(
        simplified_ui=args.simplified
    )
    
    # Run optimized system
    system = OptimizedDroneDetectionSystem(detection_config, display_config, args.log_level)
    system.run_detection_system()


if __name__ == "__main__":
    main()