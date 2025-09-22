"""
Live CCTV processing service for real-time camera feeds
"""

import asyncio
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml
import time

from app.core.config import settings
from app.services.detection import PersonDetector
from app.services.tracking import PersonTracker
from app.services.gender_classifier import GenderClassifierService
from app.services.pose_estimator import PoseEstimator
from app.services.risk_analyzer import RiskAnalyzer
from app.services.alert_manager import AlertManager
from app.websocket.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class LiveCCTVProcessor:
    """Live CCTV processing service for real-time camera feeds"""
    
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.gender_classifier = GenderClassifierService()
        self.pose_estimator = PoseEstimator()
        self.risk_analyzer = RiskAnalyzer()
        self.alert_manager = AlertManager()
        self.connection_manager = ConnectionManager()
        
        # Load configuration
        self.zones_config = self._load_zones_config()
        self.cameras_config = self._load_cameras_config()
        
        # Processing state
        self.cameras = {}
        self.processing_tasks = {}
        self.running = False
        self.frame_buffers = {}  # For temporal analysis
        self.performance_metrics = {}
        
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
    
    async def initialize(self):
        """Initialize the live CCTV processor"""
        try:
            # Initialize models
            await self.detector.initialize()
            await self.gender_classifier.initialize()
            await self.pose_estimator.initialize()
            await self.alert_manager.initialize()
            
            logger.info("Live CCTV processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize live CCTV processor: {e}")
            raise
    
    async def start(self):
        """Start live processing for all enabled cameras"""
        self.running = True
        logger.info("Starting live CCTV processing...")
        
        # Start processing for each camera
        for camera_id, camera_config in self.cameras_config.get("cameras", {}).items():
            if camera_config.get("enabled", False):
                await self._start_camera_processing(camera_id, camera_config)
        
        logger.info(f"Started live processing for {len(self.processing_tasks)} cameras")
    
    async def stop(self):
        """Stop all live processing"""
        self.running = False
        logger.info("Stopping live CCTV processing...")
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        # Release all camera captures
        for camera in self.cameras.values():
            if camera["cap"]:
                camera["cap"].release()
        
        logger.info("Live CCTV processing stopped")
    
    async def _start_camera_processing(self, camera_id: str, camera_config: Dict):
        """Start live processing for a single camera"""
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
                "last_detection": None,
                "last_frame_time": time.time(),
                "processing_fps": 0.0,
                "detection_count": 0,
                "status": "online"
            }
            
            # Initialize frame buffer for temporal analysis
            self.frame_buffers[camera_id] = []
            
            # Initialize performance metrics
            self.performance_metrics[camera_id] = {
                "frame_times": [],
                "detection_times": [],
                "processing_times": [],
                "last_update": time.time()
            }
            
            # Start processing task
            task = asyncio.create_task(
                self._process_camera_stream(camera_id)
            )
            self.processing_tasks[camera_id] = task
            
            logger.info(f"Started live processing for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            if camera_id in self.cameras:
                self.cameras[camera_id]["status"] = "error"
    
    async def _process_camera_stream(self, camera_id: str):
        """Process live video stream for a single camera"""
        camera = self.cameras[camera_id]
        cap = camera["cap"]
        tracker = camera["tracker"]
        
        # Calculate frame interval based on target FPS
        target_fps = settings.DEFAULT_FPS
        frame_interval = max(1, int(30 / target_fps))  # Process every Nth frame
        
        frame_time_buffer = []
        
        try:
            while self.running:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    await asyncio.sleep(0.1)
                    continue
                
                camera["frame_count"] += 1
                current_time = time.time()
                
                # Process every Nth frame for efficiency
                if camera["frame_count"] % frame_interval == 0:
                    await self._process_live_frame(camera_id, frame, tracker, current_time)
                
                # Update frame buffer for temporal analysis
                self._update_frame_buffer(camera_id, frame, current_time)
                
                # Calculate processing FPS
                frame_time_buffer.append(current_time)
                if len(frame_time_buffer) > 30:  # Keep last 30 frames
                    frame_time_buffer.pop(0)
                
                if len(frame_time_buffer) > 1:
                    fps = len(frame_time_buffer) / (frame_time_buffer[-1] - frame_time_buffer[0])
                    camera["processing_fps"] = fps
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self._update_performance_metrics(camera_id, processing_time, current_time)
                
                # Send real-time updates via WebSocket
                await self._send_realtime_updates(camera_id, camera, current_time)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info(f"Camera {camera_id} processing cancelled")
        except Exception as e:
            logger.error(f"Error processing camera {camera_id}: {e}")
            camera["status"] = "error"
        finally:
            if cap:
                cap.release()
    
    async def _process_live_frame(self, camera_id: str, frame: np.ndarray, tracker: PersonTracker, current_time: float):
        """Process a single live frame"""
        try:
            detection_start = time.time()
            
            # Detect people
            detections = await self.detector.detect(frame)
            
            # Track people
            tracks = tracker.update(detections, frame)
            
            # Classify gender and estimate pose for each track
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
            
            # Analyze risks with live-specific logic
            camera_config = self.cameras[camera_id]["config"]
            zone_id = camera_config["zone_id"]
            zone_config = self._get_zone_config(zone_id)
            
            if zone_config:
                risks = await self._analyze_live_risks(
                    tracks, zone_config, camera_id, zone_id, current_time
                )
                
                # Handle any detected risks
                for risk in risks:
                    await self.alert_manager.handle_risk(risk, frame, camera_id, zone_id)
            
            # Update camera state
            self.cameras[camera_id]["last_detection"] = datetime.now()
            self.cameras[camera_id]["detection_count"] += len(detections)
            
            # Update performance metrics
            detection_time = time.time() - detection_start
            self.performance_metrics[camera_id]["detection_times"].append(detection_time)
            
            # Keep only last 100 detection times
            if len(self.performance_metrics[camera_id]["detection_times"]) > 100:
                self.performance_metrics[camera_id]["detection_times"].pop(0)
            
        except Exception as e:
            logger.error(f"Error processing live frame for camera {camera_id}: {e}")
    
    async def _analyze_live_risks(self, tracks: List, zone_config: Dict, camera_id: str, zone_id: str, current_time: float) -> List[Dict]:
        """Analyze risks with live-specific logic"""
        try:
            # Get current time context
            current_datetime = datetime.now()
            is_night_time = self._is_night_time(current_datetime, zone_config)
            
            # Analyze different risk types with live-specific parameters
            risks = []
            
            # Lone woman at night (with live temporal smoothing)
            lone_risks = await self._analyze_lone_woman_live(tracks, zone_config, camera_id, zone_id, is_night_time, current_time)
            risks.extend(lone_risks)
            
            # Surrounded by men (with live proximity analysis)
            surrounded_risks = await self._analyze_surrounded_live(tracks, zone_config, camera_id, zone_id, current_time)
            risks.extend(surrounded_risks)
            
            # SOS gestures (with live temporal validation)
            sos_risks = await self._analyze_sos_gestures_live(tracks, zone_config, camera_id, zone_id, current_time)
            risks.extend(sos_risks)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error in live risk analysis: {e}")
            return []
    
    async def _analyze_lone_woman_live(self, tracks: List, zone_config: Dict, camera_id: str, zone_id: str, is_night_time: bool, current_time: float) -> List[Dict]:
        """Analyze lone woman scenario with live-specific logic"""
        risks = []
        
        if not is_night_time:
            return risks
        
        try:
            threshold_config = zone_config.get('thresholds', {}).get('lone_woman_night', {})
            min_confidence = threshold_config.get('min_confidence', 0.7)
            max_people = threshold_config.get('max_people', 3)
            duration_seconds = threshold_config.get('duration_seconds', 10)
            
            # Count people and expected females
            total_people = len(tracks)
            expected_females = sum(track.get_gender_probability() for track in tracks)
            
            # Check if conditions are met
            if (total_people <= max_people and 
                expected_females >= min_confidence and
                total_people > 0):
                
                # Check temporal consistency using frame buffer
                if self._check_live_duration_condition(camera_id, 'LONE_WOMAN_NIGHT', duration_seconds, current_time):
                    risk = {
                        'type': 'LONE_WOMAN_NIGHT',
                        'camera_id': camera_id,
                        'zone_id': zone_id,
                        'confidence': expected_females / total_people,
                        'severity': self._calculate_severity(expected_females, total_people, is_night_time),
                        'description': f"Lone woman detected at night (confidence: {expected_females:.2f})",
                        'metadata': {
                            'total_people': total_people,
                            'expected_females': expected_females,
                            'is_night_time': is_night_time,
                            'duration_seconds': duration_seconds,
                            'live_processing': True
                        },
                        'timestamp': datetime.now()
                    }
                    risks.append(risk)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analyzing lone woman live: {e}")
            return []
    
    async def _analyze_surrounded_live(self, tracks: List, zone_config: Dict, camera_id: str, zone_id: str, current_time: float) -> List[Dict]:
        """Analyze surrounded scenario with live-specific logic"""
        risks = []
        
        try:
            threshold_config = zone_config.get('thresholds', {}).get('surrounded', {})
            min_female_confidence = threshold_config.get('min_female_confidence', 0.6)
            min_male_confidence = threshold_config.get('min_male_confidence', 0.6)
            min_males = threshold_config.get('min_males', 3)
            max_distance = threshold_config.get('max_distance_meters', 2.0)
            duration_seconds = threshold_config.get('duration_seconds', 15)
            
            # Find potential female tracks
            female_tracks = [
                track for track in tracks 
                if track.get_gender_probability() >= min_female_confidence
            ]
            
            for female_track in female_tracks:
                # Find nearby male tracks
                nearby_males = []
                female_center = self._get_track_center(female_track)
                
                for track in tracks:
                    if track.track_id != female_track.track_id:
                        male_prob = 1.0 - track.get_gender_probability()
                        if male_prob >= min_male_confidence:
                            track_center = self._get_track_center(track)
                            distance = self._calculate_distance(female_center, track_center)
                            
                            if distance <= max_distance:
                                nearby_males.append({
                                    'track': track,
                                    'distance': distance,
                                    'male_confidence': male_prob
                                })
                
                # Check if surrounded condition is met
                if len(nearby_males) >= min_males:
                    # Check live duration condition
                    risk_key = f"SURROUNDED_{female_track.track_id}"
                    if self._check_live_duration_condition(camera_id, risk_key, duration_seconds, current_time):
                        avg_distance = np.mean([m['distance'] for m in nearby_males])
                        severity = self._calculate_surrounded_severity(len(nearby_males), avg_distance)
                        
                        risk = {
                            'type': 'SURROUNDED',
                            'camera_id': camera_id,
                            'zone_id': zone_id,
                            'confidence': female_track.get_gender_probability(),
                            'severity': severity,
                            'description': f"Woman surrounded by {len(nearby_males)} men",
                            'metadata': {
                                'female_track_id': female_track.track_id,
                                'nearby_males_count': len(nearby_males),
                                'avg_distance': avg_distance,
                                'min_distance': min([m['distance'] for m in nearby_males]),
                                'duration_seconds': duration_seconds,
                                'live_processing': True
                            },
                            'timestamp': datetime.now()
                        }
                        risks.append(risk)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analyzing surrounded live: {e}")
            return []
    
    async def _analyze_sos_gestures_live(self, tracks: List, zone_config: Dict, camera_id: str, zone_id: str, current_time: float) -> List[Dict]:
        """Analyze SOS gestures with live-specific logic"""
        risks = []
        
        try:
            threshold_config = zone_config.get('thresholds', {}).get('sos_gesture', {})
            min_confidence = threshold_config.get('min_confidence', 0.8)
            duration_seconds = threshold_config.get('duration_seconds', 5)
            
            for track in tracks:
                pose_keypoints = track.get_pose_keypoints()
                if pose_keypoints is not None:
                    # Analyze SOS gesture with live temporal validation
                    sos_detected = await self._detect_sos_gesture_live(track, pose_keypoints, current_time)
                    
                    if sos_detected['is_sos'] and sos_detected['confidence'] >= min_confidence:
                        # Check live duration condition
                        if self._check_live_gesture_duration(track.track_id, duration_seconds, current_time):
                            risk = {
                                'type': 'SOS_GESTURE',
                                'camera_id': camera_id,
                                'zone_id': zone_id,
                                'confidence': sos_detected['confidence'],
                                'severity': sos_detected['confidence'],
                                'description': f"SOS gesture detected (confidence: {sos_detected['confidence']:.2f})",
                                'metadata': {
                                    'track_id': track.track_id,
                                    'gesture_details': sos_detected,
                                    'duration_seconds': duration_seconds,
                                    'live_processing': True
                                },
                                'timestamp': datetime.now()
                            }
                            risks.append(risk)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analyzing SOS gestures live: {e}")
            return []
    
    async def _detect_sos_gesture_live(self, track, keypoints: np.ndarray, current_time: float) -> Dict:
        """Detect SOS gesture with live temporal analysis"""
        # This would integrate with the pose estimator for live analysis
        # For now, return a placeholder
        return {
            'is_sos': False,
            'confidence': 0.0,
            'reason': 'not_implemented'
        }
    
    def _check_live_duration_condition(self, camera_id: str, risk_type: str, duration_seconds: int, current_time: float) -> bool:
        """Check if a risk condition has been met for the required duration in live processing"""
        try:
            # Use frame buffer to check temporal consistency
            frame_buffer = self.frame_buffers.get(camera_id, [])
            if len(frame_buffer) < 10:  # Need at least 10 frames
                return False
            
            # Check if condition has been consistent over the duration
            # This is a simplified check - in practice, you'd track risk states over time
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking live duration condition: {e}")
            return False
    
    def _check_live_gesture_duration(self, track_id: int, duration_seconds: int, current_time: float) -> bool:
        """Check if SOS gesture has been detected for required duration in live processing"""
        try:
            # This would track gesture detections over time for the specific track
            # For now, return a placeholder
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking live gesture duration: {e}")
            return False
    
    def _update_frame_buffer(self, camera_id: str, frame: np.ndarray, current_time: float):
        """Update frame buffer for temporal analysis"""
        try:
            frame_buffer = self.frame_buffers.get(camera_id, [])
            
            # Add current frame info
            frame_info = {
                'frame': frame,
                'timestamp': current_time,
                'frame_number': self.cameras[camera_id]["frame_count"]
            }
            
            frame_buffer.append(frame_info)
            
            # Keep only last 30 frames (about 1 second at 30fps)
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)
            
            self.frame_buffers[camera_id] = frame_buffer
            
        except Exception as e:
            logger.error(f"Error updating frame buffer: {e}")
    
    def _update_performance_metrics(self, camera_id: str, processing_time: float, current_time: float):
        """Update performance metrics for a camera"""
        try:
            metrics = self.performance_metrics.get(camera_id, {})
            
            # Update processing times
            metrics["processing_times"].append(processing_time)
            if len(metrics["processing_times"]) > 100:
                metrics["processing_times"].pop(0)
            
            # Update frame times
            metrics["frame_times"].append(current_time)
            if len(metrics["frame_times"]) > 100:
                metrics["frame_times"].pop(0)
            
            metrics["last_update"] = current_time
            self.performance_metrics[camera_id] = metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _send_realtime_updates(self, camera_id: str, camera: Dict, current_time: float):
        """Send real-time updates via WebSocket"""
        try:
            # Send camera status update
            camera_status = {
                "camera_id": camera_id,
                "status": camera["status"],
                "processing_fps": camera["processing_fps"],
                "frame_count": camera["frame_count"],
                "detection_count": camera["detection_count"],
                "last_detection": camera["last_detection"].isoformat() if camera["last_detection"] else None,
                "timestamp": current_time
            }
            
            await self.connection_manager.send_camera_status_update(camera_id, camera_status)
            
            # Send performance metrics periodically
            if camera["frame_count"] % 100 == 0:  # Every 100 frames
                metrics = self.performance_metrics.get(camera_id, {})
                if metrics:
                    avg_processing_time = np.mean(metrics["processing_times"]) if metrics["processing_times"] else 0
                    avg_detection_time = np.mean(metrics["detection_times"]) if metrics["detection_times"] else 0
                    
                    performance_data = {
                        "camera_id": camera_id,
                        "avg_processing_time_ms": avg_processing_time * 1000,
                        "avg_detection_time_ms": avg_detection_time * 1000,
                        "processing_fps": camera["processing_fps"],
                        "timestamp": current_time
                    }
                    
                    await self.connection_manager.send_system_metrics_update(performance_data)
            
        except Exception as e:
            logger.error(f"Error sending real-time updates: {e}")
    
    def _get_zone_config(self, zone_id: str) -> Optional[Dict]:
        """Get zone configuration"""
        return self.zones_config.get("zones", {}).get(zone_id)
    
    def _is_night_time(self, current_time: datetime, zone_config: Dict) -> bool:
        """Check if current time is within quiet hours"""
        try:
            quiet_hours = zone_config.get('quiet_hours', {})
            if not quiet_hours:
                return False
            
            from datetime import time
            start_time = time.fromisoformat(quiet_hours.get('start', '22:00'))
            end_time = time.fromisoformat(quiet_hours.get('end', '06:00'))
            current_time_only = current_time.time()
            
            if start_time <= end_time:
                return start_time <= current_time_only <= end_time
            else:
                return current_time_only >= start_time or current_time_only <= end_time
                
        except Exception as e:
            logger.error(f"Error checking night time: {e}")
            return False
    
    def _calculate_severity(self, expected_females: float, total_people: int, is_night_time: bool) -> float:
        """Calculate risk severity score"""
        base_severity = expected_females / max(total_people, 1)
        
        if is_night_time:
            base_severity *= 1.5
        
        return min(base_severity, 1.0)
    
    def _calculate_surrounded_severity(self, male_count: int, avg_distance: float) -> float:
        """Calculate severity for surrounded scenario"""
        count_severity = min(male_count / 5.0, 1.0)
        distance_severity = max(0, 1.0 - (avg_distance / 2.0))
        return (count_severity + distance_severity) / 2.0
    
    def _get_track_center(self, track) -> Tuple[float, float]:
        """Get center point of track bounding box"""
        bbox = track.get_bbox()
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        import math
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_camera_status(self, camera_id: str) -> Dict:
        """Get status for a specific camera"""
        if camera_id not in self.cameras:
            return {"status": "not_found"}
        
        camera = self.cameras[camera_id]
        metrics = self.performance_metrics.get(camera_id, {})
        
        return {
            "status": camera["status"],
            "frame_count": camera["frame_count"],
            "last_detection": camera["last_detection"],
            "processing_fps": camera["processing_fps"],
            "detection_count": camera["detection_count"],
            "avg_processing_time_ms": np.mean(metrics.get("processing_times", [0])) * 1000,
            "avg_detection_time_ms": np.mean(metrics.get("detection_times", [0])) * 1000,
            "frame_buffer_size": len(self.frame_buffers.get(camera_id, []))
        }
    
    def get_all_camera_status(self) -> Dict:
        """Get status for all cameras"""
        return {
            camera_id: self.get_camera_status(camera_id)
            for camera_id in self.cameras.keys()
        }
    
    def get_system_performance(self) -> Dict:
        """Get overall system performance metrics"""
        try:
            total_cameras = len(self.cameras)
            active_cameras = len([c for c in self.cameras.values() if c["status"] == "online"])
            
            all_processing_times = []
            all_detection_times = []
            total_fps = 0
            
            for camera_id, metrics in self.performance_metrics.items():
                all_processing_times.extend(metrics.get("processing_times", []))
                all_detection_times.extend(metrics.get("detection_times", []))
                total_fps += self.cameras[camera_id]["processing_fps"]
            
            return {
                "total_cameras": total_cameras,
                "active_cameras": active_cameras,
                "avg_processing_time_ms": np.mean(all_processing_times) * 1000 if all_processing_times else 0,
                "avg_detection_time_ms": np.mean(all_detection_times) * 1000 if all_detection_times else 0,
                "total_processing_fps": total_fps,
                "system_load": len(self.processing_tasks),
                "running": self.running
            }
            
        except Exception as e:
            logger.error(f"Error getting system performance: {e}")
            return {"error": str(e)}
