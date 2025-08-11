#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import argparse
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from ultralytics import YOLO
import threading
from collections import deque

# ==================== CONFIG ====================
@dataclass
class DetectionConfig:
    model_path: str = 'best.pt'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    target_classes: List[str] = None
    auto_save_detections: bool = False
    save_directory: str = "detections"
    detection_interval: int = 1
    use_threading: bool = True
    max_queue_size: int = 2
    use_gpu_optimization: bool = True
    half_precision: bool = True

    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = ['drone']

@dataclass
class DisplayConfig:
    window_name: str = 'TEERATHAP INDUSTRY - Real-Time Drone Detection v2.1'
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False
    primary_color: Tuple[int, int, int] = (0, 255, 0)
    secondary_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (0, 165, 255)
    alert_color: Tuple[int, int, int] = (0, 0, 255)
    box_thickness: int = 2
    font_scale: float = 0.6
    simplified_ui: bool = True

class RealTimeLogger:
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

# ==================== MAIN SYSTEM ====================
class RealTimeDroneDetectionSystem:
    def __init__(self, detection_config: DetectionConfig, display_config: DisplayConfig, log_level: str = 'WARNING'):
        self.detection_config = detection_config
        self.display_config = display_config
        self.logger = RealTimeLogger(log_level).get_logger()
        self.model = None
        self.cap = None
        self.frame_count = 0
        self.last_detections = []
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()

        if self.detection_config.use_threading:
            self.detection_queue = deque(maxlen=detection_config.max_queue_size)
            self.result_queue = deque(maxlen=detection_config.max_queue_size)
            self.detection_thread = None
            self.thread_running = False

        self.session_stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'detection_count': 0
        }

        os.makedirs(self.detection_config.save_directory, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def initialize_model(self) -> bool:
        try:
            model_path = Path(self.detection_config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.logger.info("Loading YOLO model...")
            self.model = YOLO(str(model_path))
            self.model.overrides['verbose'] = False
            if self.detection_config.use_gpu_optimization:
                self.model.overrides['device'] = 'cuda'
            if self.detection_config.half_precision:
                try:
                    self.model.model.half()
                    self.logger.info("Half precision enabled")
                except:
                    self.logger.warning("Half precision not supported")

            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)

            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def initialize_camera(self, camera_source) -> bool:
        """ รองรับทั้ง Webcam (int) และ IP Camera (string URL) """
        try:
            self.logger.info(f"Initializing camera: {camera_source} ...")

            # แยกประเภท input
            if isinstance(camera_source, int) or (isinstance(camera_source, str) and camera_source.isdigit()):
                camera_source = int(camera_source)

            self.cap = cv2.VideoCapture(camera_source)

            if not self.cap.isOpened():
                raise IOError("Cannot access camera stream")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Camera ready: {width}x{height} @ {fps} FPS")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False

    def detection_worker(self):
        while self.thread_running:
            if len(self.detection_queue) > 0:
                frame = self.detection_queue.popleft()
                start_time = time.time()
                results = self.model(frame,
                                     conf=self.detection_config.confidence_threshold,
                                     iou=self.detection_config.iou_threshold,
                                     max_det=self.detection_config.max_detections,
                                     verbose=False)
                detections = self.process_detections_fast(results)
                detection_time = time.time() - start_time
                self.result_queue.clear()
                self.result_queue.append({
                    'detections': detections,
                    'processing_time': detection_time
                })
            else:
                time.sleep(0.0001)

    def process_detections_fast(self, results) -> List[Dict]:
        detections = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        conf_mask = confidences >= self.detection_config.confidence_threshold
        valid_boxes = boxes[conf_mask]
        valid_confs = confidences[conf_mask]
        valid_classes = classes[conf_mask]
        target_set = {cls.lower() for cls in self.detection_config.target_classes}
        for i in range(len(valid_boxes)):
            cls_id = valid_classes[i]
            class_name = self.model.names[cls_id]
            if class_name.lower() in target_set:
                x1, y1, x2, y2 = valid_boxes[i].astype(int)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(valid_confs[i]),
                    'class_name': class_name
                })
        return detections

    def calculate_fps(self) -> float:
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return (len(self.fps_counter) - 1) / time_diff
        return 0.0

    def run_detection_system(self, camera_source):
        if not self.initialize_model():
            return
        if not self.initialize_camera(camera_source):
            return
        cv2.namedWindow(self.display_config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_config.window_name,
                         self.display_config.window_width,
                         self.display_config.window_height)
        if self.detection_config.use_threading:
            self.thread_running = True
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()

        detection_times = deque(maxlen=30)

        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    continue
                self.frame_count += 1
                self.session_stats['total_frames'] += 1
                if self.detection_config.use_threading:
                    self.detection_queue.clear()
                    self.detection_queue.append(frame.copy())
                    if len(self.result_queue) > 0:
                        result_data = self.result_queue.popleft()
                        self.last_detections = result_data['detections']
                        detection_times.append(result_data['processing_time'])
                        self.session_stats['detection_count'] += len(self.last_detections)
                    detections = self.last_detections
                else:
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

                fps = self.calculate_fps()
                avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                frame = self.draw_minimal_overlay(frame, detections, fps, avg_detection_time)
                cv2.imshow(self.display_config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            self.cleanup()

    def draw_minimal_overlay(self, frame, detections, fps, detection_time):
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        status_text = f"REAL-TIME | FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms | Targets: {len(detections)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{detection['confidence']:.0%}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame

    def cleanup(self):
        if self.detection_config.use_threading and self.thread_running:
            self.thread_running = False
            if self.detection_thread:
                self.detection_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# ==================== CLI ====================
def create_parser():
    parser = argparse.ArgumentParser(description="Real-Time Drone Detection System")
    parser.add_argument('--model', '-m', default='best.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--camera', default=0, help='Camera device ID or RTSP/HTTP URL')
    parser.add_argument('--classes', nargs='+', default=['drone'], help='Target classes')
    parser.add_argument('--no-threading', action='store_true', help='Disable threading')
    parser.add_argument('--half-precision', action='store_true', help='Enable FP16 for speed boost')
    parser.add_argument('--gpu', action='store_true', help='Force GPU acceleration')
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    detection_config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        target_classes=args.classes,
        detection_interval=1,
        use_threading=not args.no_threading,
        auto_save_detections=False,
        use_gpu_optimization=args.gpu,
        half_precision=args.half_precision
    )
    display_config = DisplayConfig()
    system = RealTimeDroneDetectionSystem(detection_config, display_config, args.log_level)
    system.run_detection_system(args.camera)

if __name__ == "__main__":
    main()
