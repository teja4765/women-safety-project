"""
Person detection service using YOLO
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PersonDetector:
    """Person detection using YOLO"""
    
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
    async def initialize(self):
        """Initialize the YOLO model"""
        try:
            # Load YOLO model (will download if not present)
            self.model = YOLO('yolov8n.pt')  # Nano version for speed
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise
    
    async def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, class
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only keep person detections (class 0 in COCO)
                        if class_id == 0 and confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': 'person'
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Detection confidence threshold set to {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": "YOLOv8n",
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_size": "640x640",
            "classes": ["person"]
        }
