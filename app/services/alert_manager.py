"""
Alert management service for handling and storing safety alerts
"""

import asyncio
import uuid
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
import os

from app.core.config import settings
from app.services.storage import StorageService
from app.services.notification import NotificationService

logger = logging.getLogger(__name__)


class AlertManager:
    """Alert management service"""
    
    def __init__(self):
        self.storage_service = StorageService()
        self.notification_service = NotificationService()
        self.alert_cooldowns = {}  # Track cooldown periods
        self.recent_alerts = {}  # Track recent alerts to prevent spam
        
    async def initialize(self):
        """Initialize alert manager"""
        try:
            await self.storage_service.initialize()
            await self.notification_service.initialize()
            logger.info("Alert manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {e}")
            raise
    
    async def handle_risk(self, risk: Dict, frame: np.ndarray, camera_id: str, zone_id: str):
        """
        Handle a detected risk by creating alert and taking actions
        
        Args:
            risk: Risk detection result
            frame: Current video frame
            camera_id: Camera identifier
            zone_id: Zone identifier
        """
        try:
            # Check cooldown period
            if self._is_in_cooldown(camera_id, risk['type']):
                logger.debug(f"Alert for {risk['type']} in cooldown for camera {camera_id}")
                return
            
            # Check rate limiting
            if self._is_rate_limited(camera_id):
                logger.warning(f"Rate limit exceeded for camera {camera_id}")
                return
            
            # Create alert
            alert = await self._create_alert(risk, camera_id, zone_id)
            
            # Process video clip
            clip_url = await self._process_video_clip(alert['id'], frame, camera_id)
            alert['clip_url'] = clip_url
            
            # Generate thumbnail
            thumbnail_url = await self._generate_thumbnail(alert['id'], frame)
            alert['thumbnail_url'] = thumbnail_url
            
            # Store alert
            await self._store_alert(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Update cooldown and rate limiting
            self._update_cooldown(camera_id, risk['type'])
            self._update_rate_limit(camera_id)
            
            logger.info(f"Alert created: {alert['id']} - {risk['type']} at {camera_id}")
            
        except Exception as e:
            logger.error(f"Error handling risk: {e}")
    
    async def _create_alert(self, risk: Dict, camera_id: str, zone_id: str) -> Dict:
        """Create alert object from risk detection"""
        alert_id = str(uuid.uuid4())
        
        alert = {
            'id': alert_id,
            'type': risk['type'],
            'camera_id': camera_id,
            'zone_id': zone_id,
            'start_time': risk['timestamp'],
            'end_time': risk['timestamp'] + timedelta(seconds=settings.CLIP_DURATION_SECONDS),
            'confidence': risk['confidence'],
            'severity': risk['severity'],
            'description': risk['description'],
            'metadata': risk.get('metadata', {}),
            'status': 'pending',
            'created_at': datetime.now()
        }
        
        return alert
    
    async def _process_video_clip(self, alert_id: str, frame: np.ndarray, camera_id: str) -> str:
        """Process video clip with face blurring"""
        try:
            # Create output directory
            output_dir = f"data/clips/{alert_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Apply face blurring
            blurred_frame = await self._blur_faces(frame)
            
            # Save frame as image (for now, in production this would be a video clip)
            clip_path = f"{output_dir}/event.jpg"
            cv2.imwrite(clip_path, blurred_frame)
            
            # Upload to storage
            clip_url = await self.storage_service.upload_file(clip_path, f"alerts/{alert_id}/event.jpg")
            
            return clip_url
            
        except Exception as e:
            logger.error(f"Error processing video clip: {e}")
            return ""
    
    async def _generate_thumbnail(self, alert_id: str, frame: np.ndarray) -> str:
        """Generate thumbnail for alert"""
        try:
            # Resize frame for thumbnail
            height, width = frame.shape[:2]
            thumbnail_size = (320, 240)
            thumbnail = cv2.resize(frame, thumbnail_size)
            
            # Apply face blurring
            blurred_thumbnail = await self._blur_faces(thumbnail)
            
            # Save thumbnail
            output_dir = f"data/clips/{alert_id}"
            os.makedirs(output_dir, exist_ok=True)
            thumbnail_path = f"{output_dir}/thumbnail.jpg"
            cv2.imwrite(thumbnail_path, blurred_thumbnail)
            
            # Upload to storage
            thumbnail_url = await self.storage_service.upload_file(thumbnail_path, f"alerts/{alert_id}/thumbnail.jpg")
            
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
    
    async def _store_alert(self, alert: Dict):
        """Store alert in database"""
        try:
            # This would integrate with the database service
            # For now, just log the alert
            logger.info(f"Storing alert: {json.dumps(alert, default=str)}")
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _send_notifications(self, alert: Dict):
        """Send notifications for alert"""
        try:
            await self.notification_service.send_alert_notification(alert)
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _is_in_cooldown(self, camera_id: str, risk_type: str) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_key = f"{camera_id}_{risk_type}"
        if cooldown_key in self.alert_cooldowns:
            cooldown_time = self.alert_cooldowns[cooldown_key]
            if datetime.now() < cooldown_time:
                return True
        return False
    
    def _is_rate_limited(self, camera_id: str) -> bool:
        """Check if camera has exceeded rate limit"""
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)
        
        if camera_id not in self.recent_alerts:
            self.recent_alerts[camera_id] = []
        
        # Remove old alerts
        self.recent_alerts[camera_id] = [
            alert_time for alert_time in self.recent_alerts[camera_id]
            if alert_time > hour_ago
        ]
        
        # Check rate limit
        return len(self.recent_alerts[camera_id]) >= settings.MAX_ALERTS_PER_HOUR
    
    def _update_cooldown(self, camera_id: str, risk_type: str):
        """Update cooldown period for alert type"""
        cooldown_key = f"{camera_id}_{risk_type}"
        cooldown_time = datetime.now() + timedelta(seconds=settings.ALERT_COOLDOWN_SECONDS)
        self.alert_cooldowns[cooldown_key] = cooldown_time
    
    def _update_rate_limit(self, camera_id: str):
        """Update rate limiting for camera"""
        if camera_id not in self.recent_alerts:
            self.recent_alerts[camera_id] = []
        
        self.recent_alerts[camera_id].append(datetime.now())
    
    async def acknowledge_alert(self, alert_id: str, operator_id: str, feedback: Optional[str] = None):
        """Acknowledge an alert"""
        try:
            # This would update the database
            logger.info(f"Alert {alert_id} acknowledged by {operator_id}")
            if feedback:
                logger.info(f"Feedback: {feedback}")
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
    
    async def escalate_alert(self, alert_id: str, operator_id: str, escalation_reason: str):
        """Escalate an alert"""
        try:
            # This would update the database and send escalation notifications
            logger.info(f"Alert {alert_id} escalated by {operator_id}: {escalation_reason}")
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
    
    def get_alert_statistics(self, hours: int = 24) -> Dict:
        """Get alert statistics"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=hours)
            
            total_alerts = 0
            alerts_by_type = {}
            alerts_by_camera = {}
            
            # Count alerts from recent_alerts (simplified)
            for camera_id, alert_times in self.recent_alerts.items():
                recent_count = len([t for t in alert_times if t > hour_ago])
                if recent_count > 0:
                    alerts_by_camera[camera_id] = recent_count
                    total_alerts += recent_count
            
            return {
                'total_alerts': total_alerts,
                'alerts_by_type': alerts_by_type,
                'alerts_by_camera': alerts_by_camera,
                'time_window_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {'total_alerts': 0, 'alerts_by_type': {}, 'alerts_by_camera': {}, 'time_window_hours': hours}
