"""
Video processing service for real-time computer vision pipeline
"""

import asyncio
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import yaml

from app.core.config import settings
from app.services.detection import PersonDetector
from app.services.tracking import PersonTracker
from app.services.gender_classifier import GenderClassifier
from app.services.pose_estimator import PoseEstimator
from app.services.risk_analyzer import RiskAnalyzer
from app.services.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing service"""
    
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.gender_classifier = GenderClassifier()
        self.pose_estimator = PoseEstimator()
        self.risk_analyzer = RiskAnalyzer()
        self.alert_manager = AlertManager()
        
        # Load configuration
        self.zones_config = self._load_zones_config()
        self.cameras_config = self._load_cameras_config()
        
        # Processing state
        self.cameras = {}
        self.processing_tasks = {}
        self.running = False
        
    def _load_zones_config(self) -> Dict:
        """Load zones configuration"""
        try:
            with open("config/zones.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load zones config: {e}")
            return {}
    
    def _load_cameras_config(self) -> Dict:
        """Load cameras configuration"""
        try:
            with open("config/cameras.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load cameras config: {e}")
            return {}
    
    async def start(self):
        """Start video processing for all enabled cameras"""
        self.running = True
        logger.info("Starting video processing...")
        
        # Initialize models
        await self._initialize_models()
        
        # Start processing for each camera
        for camera_id, camera_config in self.cameras_config.get("cameras", {}).items():
            if camera_config.get("enabled", False):
                await self._start_camera_processing(camera_id, camera_config)
        
        logger.info(f"Started processing for {len(self.processing_tasks)} cameras")
    
    async def stop(self):
        """Stop all video processing"""
        self.running = False
        logger.info("Stopping video processing...")
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        logger.info("Video processing stopped")
    
    async def _initialize_models(self):
        """Initialize all ML models"""
        try:
            await self.detector.initialize()
            await self.gender_classifier.initialize()
            await self.pose_estimator.initialize()
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _start_camera_processing(self, camera_id: str, camera_config: Dict):
        """Start processing for a single camera"""
        try:
            # Create camera capture
            cap = cv2.VideoCapture(camera_config["source"])
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get("resolution", [1920, 1080])[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get("resolution", [1920, 1080])[1])
            cap.set(cv2.CAP_PROP_FPS, camera_config.get("fps", 30))
            
            # Store camera info
            self.cameras[camera_id] = {
                "config": camera_config,
                "cap": cap,
                "tracker": PersonTracker(),
                "frame_count": 0,
                "last_detection": None
            }
            
            # Start processing task
            task = asyncio.create_task(
                self._process_camera_stream(camera_id)
            )
            self.processing_tasks[camera_id] = task
            
            logger.info(f"Started processing for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
    
    async def _process_camera_stream(self, camera_id: str):
        """Process video stream for a single camera"""
        camera = self.cameras[camera_id]
        cap = camera["cap"]
        tracker = camera["tracker"]
        
        frame_interval = max(1, int(30 / settings.DEFAULT_FPS))  # Process every Nth frame
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    await asyncio.sleep(0.1)
                    continue
                
                camera["frame_count"] += 1
                
                # Process every Nth frame
                if camera["frame_count"] % frame_interval == 0:
                    await self._process_frame(camera_id, frame, tracker)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info(f"Camera {camera_id} processing cancelled")
        except Exception as e:
            logger.error(f"Error processing camera {camera_id}: {e}")
        finally:
            cap.release()
    
    async def _process_frame(self, camera_id: str, frame: np.ndarray, tracker: PersonTracker):
        """Process a single frame"""
        try:
            # Detect people
            detections = await self.detector.detect(frame)
            
            # Track people
            tracks = tracker.update(detections, frame)
            
            # Classify gender for each track
            for track in tracks:
                if track.is_confirmed():
                    # Extract person crop
                    bbox = track.get_bbox()
                    person_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    if person_crop.size > 0:
                        # Classify gender
                        gender_prob = await self.gender_classifier.classify(person_crop)
                        track.set_gender_probability(gender_prob)
                        
                        # Estimate pose for SOS detection
                        pose_keypoints = await self.pose_estimator.estimate(person_crop)
                        track.set_pose_keypoints(pose_keypoints)
            
            # Analyze risks
            camera_config = self.cameras[camera_id]["config"]
            zone_id = camera_config["zone_id"]
            zone_config = self._get_zone_config(zone_id)
            
            if zone_config:
                risks = await self.risk_analyzer.analyze_risks(
                    tracks, zone_config, camera_id, zone_id
                )
                
                # Handle any detected risks
                for risk in risks:
                    await self.alert_manager.handle_risk(risk, frame, camera_id, zone_id)
            
            # Update camera state
            self.cameras[camera_id]["last_detection"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
    
    def _get_zone_config(self, zone_id: str) -> Optional[Dict]:
        """Get zone configuration"""
        return self.zones_config.get("zones", {}).get(zone_id)
    
    def get_camera_status(self, camera_id: str) -> Dict:
        """Get status for a specific camera"""
        if camera_id not in self.cameras:
            return {"status": "not_found"}
        
        camera = self.cameras[camera_id]
        return {
            "status": "running" if self.running else "stopped",
            "frame_count": camera["frame_count"],
            "last_detection": camera["last_detection"],
            "processing_fps": settings.DEFAULT_FPS
        }
    
    def get_all_camera_status(self) -> Dict:
        """Get status for all cameras"""
        return {
            camera_id: self.get_camera_status(camera_id)
            for camera_id in self.cameras.keys()
        }
