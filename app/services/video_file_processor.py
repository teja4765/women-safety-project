"""
Video file processing service for batch analysis
"""

import asyncio
import cv2
import numpy as np
import os
import uuid
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import yaml
from pathlib import Path

from app.core.config import settings
from app.services.detection import PersonDetector
from app.services.tracking import PersonTracker
from app.services.gender_classifier import GenderClassifierService
from app.services.pose_estimator import PoseEstimator
from app.services.risk_analyzer import RiskAnalyzer
from app.services.alert_manager import AlertManager
from app.services.storage import StorageService
from app.services.violence_classifier import ViolenceClassifierService

logger = logging.getLogger(__name__)


class VideoFileProcessor:
    """Video file processing service for batch analysis"""
    
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.gender_classifier = GenderClassifierService()
        self.pose_estimator = PoseEstimator()
        self.risk_analyzer = RiskAnalyzer()
        self.alert_manager = AlertManager()
        self.storage_service = StorageService()
        self.violence_classifier = ViolenceClassifierService()
        
        # Load configuration
        self.zones_config = self._load_zones_config()
        
        # Processing state
        self.processing_jobs = {}
        self.completed_jobs = {}
        
    def _load_zones_config(self) -> Dict:
        """Load zones configuration"""
        try:
            with open("config/zones.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load zones config: {e}")
            return {}
    
    async def initialize(self):
        """Initialize the video file processor"""
        try:
            # Initialize models
            await self.detector.initialize()
            await self.gender_classifier.initialize()
            await self.pose_estimator.initialize()
            await self.alert_manager.initialize()
            await self.storage_service.initialize()
            await self.violence_classifier.initialize()
            
            logger.info("Video file processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video file processor: {e}")
            raise
    
    async def process_video_file(
        self, 
        video_path: str, 
        zone_id: str, 
        camera_id: str = None,
        processing_mode: str = "batch",
        output_dir: str = None
    ) -> Dict:
        """
        Process a video file for safety analysis
        
        Args:
            video_path: Path to the video file
            zone_id: Zone ID for analysis context
            camera_id: Camera ID (optional, will be generated if not provided)
            processing_mode: "batch" or "realtime"
            output_dir: Output directory for results
            
        Returns:
            Processing job information
        """
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            if not camera_id:
                camera_id = f"video_file_{job_id[:8]}"
            
            if not output_dir:
                output_dir = f"data/processed_videos/{job_id}"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize job
            job_info = {
                "job_id": job_id,
                "video_path": video_path,
                "zone_id": zone_id,
                "camera_id": camera_id,
                "processing_mode": processing_mode,
                "output_dir": output_dir,
                "status": "initializing",
                "progress": 0.0,
                "start_time": datetime.now(),
                "end_time": None,
                "total_frames": 0,
                "processed_frames": 0,
                "alerts_found": 0,
                "errors": []
            }
            
            self.processing_jobs[job_id] = job_info
            
            # Start processing task
            task = asyncio.create_task(
                self._process_video_async(job_id)
            )
            
            logger.info(f"Started video processing job {job_id} for {video_path}")
            return job_info
            
        except Exception as e:
            logger.error(f"Error starting video processing: {e}")
            raise
    
    async def _process_video_async(self, job_id: str):
        """Process video file asynchronously"""
        try:
            job_info = self.processing_jobs[job_id]
            job_info["status"] = "processing"
            
            # Open video file
            cap = cv2.VideoCapture(job_info["video_path"])
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {job_info['video_path']}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            job_info["total_frames"] = total_frames
            job_info["fps"] = fps
            job_info["duration"] = duration
            
            # Get zone configuration
            zone_config = self.zones_config.get("zones", {}).get(job_info["zone_id"])
            if not zone_config:
                logger.warning(f"No zone configuration found for {job_info['zone_id']}")
                zone_config = self._get_default_zone_config()
            
            # Initialize tracking and analysis
            tracker = PersonTracker()
            frame_buffer = []
            alerts_found = 0
            # Stats aggregation
            total_person_detections = 0
            frames_with_persons = 0
            total_female_like = 0
            total_male_like = 0
            max_concurrent_people = 0
            last_people = 0
            last_females = 0
            last_males = 0
            
            # Process frames
            frame_count = 0
            clip_frames = []  # collect frames for violence classifier
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                try:
                    # Detect people
                    detections = await self.detector.detect(frame)
                    
                    # Track people
                    tracks = tracker.update(detections, frame)
                    
                    # Classify gender and estimate pose for each track
                    for track in tracks:
                        if track.is_confirmed():
                            bbox = track.get_bbox()
                            person_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            
                            if person_crop.size > 0:
                                # Classify gender
                                gender_prob = await self.gender_classifier.classify(person_crop)
                                track.set_gender_probability(gender_prob)
                                
                                # Estimate pose
                                pose_keypoints = await self.pose_estimator.estimate(person_crop)
                                track.set_pose_keypoints(pose_keypoints)
                    
                    # Prepare violence clip (collect downsampled frames)
                    if frame_count % max(1, int(fps / max(1,  self.violence_classifier.fps))) == 0:
                        resized = cv2.resize(frame, (self.violence_classifier.input_size, self.violence_classifier.input_size))
                        clip_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                        if len(clip_frames) > self.violence_classifier.input_frames:
                            clip_frames.pop(0)

                    violent_prob = 0.0
                    violent_flag = False
                    if len(clip_frames) == self.violence_classifier.input_frames:
                        violent_prob = self.violence_classifier.predict_prob(np.stack(clip_frames, axis=0))
                        violent_flag = self.violence_classifier.is_violent(violent_prob)

                    # Analyze risks
                    risks = await self.risk_analyzer.analyze_risks(
                        tracks, zone_config, job_info["camera_id"], job_info["zone_id"]
                    )
                    
                    # Handle detected risks
                    for risk in risks:
                        await self._handle_video_risk(risk, frame, job_info, frame_count)
                        alerts_found += 1
                    
                    # Aggregate stats
                    confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                    num_people = len(confirmed_tracks)
                    total_person_detections += num_people
                    if num_people > 0:
                        frames_with_persons += 1
                    fem_ct = 0
                    male_ct = 0
                    for t in confirmed_tracks:
                        gp = t.get_gender_probability()
                        if gp is not None:
                            if gp >= 0.5:
                                total_female_like += 1
                                fem_ct += 1
                            else:
                                total_male_like += 1
                                male_ct += 1
                    # update concurrent stats
                    last_people = num_people
                    last_females = fem_ct
                    last_males = male_ct
                    if num_people > max_concurrent_people:
                        max_concurrent_people = num_people

                    # Store frame with annotations for review
                    if risks or job_info["processing_mode"] == "batch":
                        annotated_frame = self._annotate_frame(frame, tracks, risks)
                        # Overlay violence probability
                        if violent_prob > 0:
                            text = f"Violence: {violent_prob:.2f}"
                            color = (0, 0, 255) if violent_flag else (0, 165, 255)
                            cv2.putText(annotated_frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        frame_path = os.path.join(
                            job_info["output_dir"], 
                            f"frame_{frame_count:06d}.jpg"
                        )
                        cv2.imwrite(frame_path, annotated_frame)
                    
                    # Update progress
                    progress = (frame_count / total_frames) * 100
                    job_info["progress"] = progress
                    job_info["processed_frames"] = frame_count
                    job_info["alerts_found"] = alerts_found
                    # Save intermittent violence stats
                    if violent_prob > 0:
                        vf = job_info.setdefault("violence_stats", {"frames_flagged": 0, "samples": []})
                        if violent_flag:
                            vf["frames_flagged"] += 1
                        if frame_count % 10 == 0:
                            vf["samples"].append({"frame": frame_count, "prob": float(violent_prob)})
                    
                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        logger.info(f"Job {job_id}: Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                    
                except Exception as e:
                    error_msg = f"Error processing frame {frame_count}: {e}"
                    logger.error(error_msg)
                    job_info["errors"].append(error_msg)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            # Finalize job
            cap.release()
            job_info["status"] = "completed"
            job_info["end_time"] = datetime.now()
            job_info["processing_time"] = (job_info["end_time"] - job_info["start_time"]).total_seconds()
            # Finalize stats
            job_info["detection_stats"] = {
                "total_person_detections": total_person_detections,
                "frames_with_persons": frames_with_persons,
                "estimated_female_count": total_female_like,
                "estimated_male_count": total_male_like,
                "avg_people_per_frame": (total_person_detections / max(1, job_info["processed_frames"])),
                "max_concurrent_people": max_concurrent_people,
                "last_people": last_people,
                "last_females": last_females,
                "last_males": last_males
            }
            vs = job_info.get("violence_stats", {"frames_flagged": 0, "samples": []})
            job_info["violence_stats"] = vs
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.processing_jobs[job_id]
            
            # Generate summary report
            await self._generate_processing_report(job_info)
            
            logger.info(f"Completed video processing job {job_id}: {alerts_found} alerts found")
            
        except Exception as e:
            error_msg = f"Error in video processing job {job_id}: {e}"
            logger.error(error_msg)
            
            if job_id in self.processing_jobs:
                job_info = self.processing_jobs[job_id]
                job_info["status"] = "failed"
                job_info["end_time"] = datetime.now()
                job_info["errors"].append(error_msg)
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job_info
                del self.processing_jobs[job_id]
    
    async def _handle_video_risk(self, risk: Dict, frame: np.ndarray, job_info: Dict, frame_number: int):
        """Handle risk detected in video file"""
        try:
            # Create alert with video-specific metadata
            alert = {
                'id': f"{job_info['job_id']}_{frame_number}_{risk['type']}",
                'type': risk['type'],
                'camera_id': job_info['camera_id'],
                'zone_id': job_info['zone_id'],
                'start_time': datetime.now(),
                'end_time': datetime.now(),
                'confidence': risk['confidence'],
                'severity': risk['severity'],
                'description': f"{risk['description']} (Frame {frame_number})",
                'metadata': {
                    **risk.get('metadata', {}),
                    'video_file': job_info['video_path'],
                    'frame_number': frame_number,
                    'job_id': job_info['job_id'],
                    'processing_mode': job_info['processing_mode']
                },
                'status': 'pending',
                'created_at': datetime.now()
            }
            
            # Process video clip (extract frame and surrounding context)
            clip_url = await self._process_video_clip(alert, frame, job_info, frame_number)
            alert['clip_url'] = clip_url
            
            # Generate thumbnail
            thumbnail_url = await self._generate_thumbnail(alert, frame)
            alert['thumbnail_url'] = thumbnail_url
            
            # Store alert
            await self._store_video_alert(alert)
            
            logger.info(f"Video alert created: {alert['id']} - {risk['type']} at frame {frame_number}")
            
        except Exception as e:
            logger.error(f"Error handling video risk: {e}")
    
    async def _process_video_clip(self, alert: Dict, frame: np.ndarray, job_info: Dict, frame_number: int) -> str:
        """Process video clip from frame"""
        try:
            # Create clip directory
            clip_dir = os.path.join(job_info["output_dir"], "clips", alert['id'])
            os.makedirs(clip_dir, exist_ok=True)
            
            # Apply face blurring
            blurred_frame = await self._blur_faces(frame)
            
            # Save frame as clip
            clip_path = os.path.join(clip_dir, "frame.jpg")
            cv2.imwrite(clip_path, blurred_frame)
            
            # Upload to storage
            clip_url = await self.storage_service.upload_file(
                clip_path, 
                f"video_alerts/{alert['id']}/frame.jpg"
            )
            
            return clip_url
            
        except Exception as e:
            logger.error(f"Error processing video clip: {e}")
            return ""
    
    async def _generate_thumbnail(self, alert: Dict, frame: np.ndarray) -> str:
        """Generate thumbnail for video alert"""
        try:
            # Resize frame for thumbnail
            thumbnail = cv2.resize(frame, (320, 240))
            
            # Apply face blurring
            blurred_thumbnail = await self._blur_faces(thumbnail)
            
            # Save thumbnail under the job output directory
            # job_id maps to a specific output directory recorded in completed_jobs
            job_output_dir = None
            if alert and alert.get('metadata'):
                job_id = alert['metadata'].get('job_id')
                if job_id and job_id in self.completed_jobs:
                    job_output_dir = self.completed_jobs[job_id].get('output_dir')
            if not job_output_dir:
                job_output_dir = f"data/processed_videos/{alert['metadata'].get('job_id', 'unknown')}"
            clip_dir = os.path.join(job_output_dir, "clips", alert['id'])
            os.makedirs(clip_dir, exist_ok=True)
            thumbnail_path = os.path.join(clip_dir, "thumbnail.jpg")
            cv2.imwrite(thumbnail_path, blurred_thumbnail)
            
            # Upload to storage
            thumbnail_url = await self.storage_service.upload_file(
                thumbnail_path, 
                f"video_alerts/{alert['id']}/thumbnail.jpg"
            )
            
            return thumbnail_url
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return ""
    
    async def _blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """Blur faces in frame for privacy protection"""
        try:
            # Load face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Blur detected faces
            blurred_frame = frame.copy()
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = blurred_frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_region, (settings.FACE_BLUR_RADIUS, settings.FACE_BLUR_RADIUS), 0)
                
                # Replace face region
                blurred_frame[y:y+h, x:x+w] = blurred_face
            
            return blurred_frame
            
        except Exception as e:
            logger.error(f"Error blurring faces: {e}")
            return frame
    
    async def _store_video_alert(self, alert: Dict):
        """Store video alert in database"""
        try:
            # This would integrate with the database service
            # For now, save to JSON file
            job_id = alert['metadata'].get('job_id')
            output_dir = f"data/processed_videos/{job_id}" if job_id else "data/processed_videos"
            alert_file = os.path.join(output_dir, "alerts", f"{alert['id']}.json")
            os.makedirs(os.path.dirname(alert_file), exist_ok=True)
            
            import json
            with open(alert_file, 'w') as f:
                json.dump(alert, f, default=str, indent=2)
            
            logger.info(f"Video alert stored: {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error storing video alert: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List, risks: List) -> np.ndarray:
        """Annotate frame with detection and risk information"""
        try:
            annotated_frame = frame.copy()
            
            # Draw tracks
            for track in tracks:
                if track.is_confirmed():
                    bbox = track.get_bbox()
                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green for normal
                    if track.get_gender_probability() > 0.6:
                        color = (255, 0, 255)  # Magenta for female
                    elif track.get_gender_probability() < 0.4:
                        color = (255, 0, 0)  # Blue for male
                    
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw gender probability
                    gp = track.get_gender_probability()
                    gender_text = f"F:{gp:.2f}" if gp is not None else "F:--"
                    cv2.putText(annotated_frame, gender_text, (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw risk indicators
            for risk in risks:
                risk_text = f"RISK: {risk['type']} ({risk['confidence']:.2f})"
                cv2.putText(annotated_frame, risk_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame
    
    def _get_default_zone_config(self) -> Dict:
        """Get default zone configuration"""
        return {
            "thresholds": {
                "lone_woman_night": {
                    "min_confidence": 0.7,
                    "max_people": 3,
                    "duration_seconds": 10
                },
                "surrounded": {
                    "min_female_confidence": 0.6,
                    "min_male_confidence": 0.6,
                    "min_males": 3,
                    "max_distance_meters": 2.0,
                    "duration_seconds": 15
                },
                "sos_gesture": {
                    "min_confidence": 0.8,
                    "duration_seconds": 5
                }
            },
            "quiet_hours": {
                "start": "22:00",
                "end": "06:00"
            }
        }
    
    async def _generate_processing_report(self, job_info: Dict):
        """Generate processing report for completed job"""
        try:
            report = {
                "job_id": job_info["job_id"],
                "video_path": job_info["video_path"],
                "zone_id": job_info["zone_id"],
                "camera_id": job_info["camera_id"],
                "processing_mode": job_info["processing_mode"],
                "start_time": job_info["start_time"],
                "end_time": job_info["end_time"],
                "processing_time_seconds": job_info["processing_time"],
                "total_frames": job_info["total_frames"],
                "processed_frames": job_info["processed_frames"],
                "fps": job_info.get("fps", 0),
                "duration_seconds": job_info.get("duration", 0),
                "alerts_found": job_info["alerts_found"],
                "errors": job_info["errors"],
                "output_directory": job_info["output_dir"]
            }
            
            # Save report
            report_path = os.path.join(job_info["output_dir"], "processing_report.json")
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, default=str, indent=2)
            
            logger.info(f"Processing report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating processing report: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a processing job"""
        if job_id in self.processing_jobs:
            return self.processing_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    def get_all_jobs(self) -> Dict:
        """Get all processing jobs"""
        return {
            "active": self.processing_jobs,
            "completed": self.completed_jobs
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job"""
        if job_id in self.processing_jobs:
            job_info = self.processing_jobs[job_id]
            job_info["status"] = "cancelled"
            job_info["end_time"] = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.processing_jobs[job_id]
            
            logger.info(f"Cancelled video processing job {job_id}")
            return True
        return False
