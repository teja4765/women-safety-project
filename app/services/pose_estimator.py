"""
Pose estimation service for SOS gesture detection
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Pose estimation using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Keypoint indices for relevant body parts
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.NOSE = 0
        
    async def initialize(self):
        """Initialize MediaPipe pose estimation"""
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Pose estimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            raise
    
    async def estimate(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose keypoints for a person crop
        
        Args:
            person_crop: BGR image crop of person
            
        Returns:
            Array of keypoints or None if no pose detected
        """
        if self.pose is None:
            raise RuntimeError("Pose estimator not initialized")
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.visibility])
                
                return np.array(keypoints)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return None
    
    def detect_sos_gesture(self, keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> dict:
        """
        Detect SOS gesture from pose keypoints
        
        Args:
            keypoints: Array of pose keypoints
            frame_shape: (height, width) of the frame
            
        Returns:
            Dictionary with SOS detection results
        """
        if keypoints is None or len(keypoints) < 17:
            return {
                "is_sos": False,
                "confidence": 0.0,
                "reason": "insufficient_keypoints"
            }
        
        try:
            height, width = frame_shape
            
            # Get relevant keypoints
            left_wrist = keypoints[self.LEFT_WRIST]
            right_wrist = keypoints[self.RIGHT_WRIST]
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            nose = keypoints[self.NOSE]
            
            # Check visibility
            if (left_wrist[2] < 0.5 or right_wrist[2] < 0.5 or 
                left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5):
                return {
                    "is_sos": False,
                    "confidence": 0.0,
                    "reason": "low_visibility"
                }
            
            # Convert to pixel coordinates
            left_wrist_px = [left_wrist[0] * width, left_wrist[1] * height]
            right_wrist_px = [right_wrist[0] * width, right_wrist[1] * height]
            left_shoulder_px = [left_shoulder[0] * width, left_shoulder[1] * height]
            right_shoulder_px = [right_shoulder[0] * width, right_shoulder[1] * height]
            nose_px = [nose[0] * width, nose[1] * height]
            
            # Check if hands are above shoulders
            left_hand_above = left_wrist_px[1] < min(left_shoulder_px[1], right_shoulder_px[1])
            right_hand_above = right_wrist_px[1] < min(left_shoulder_px[1], right_shoulder_px[1])
            
            # Check if hands are raised (above nose level)
            left_hand_raised = left_wrist_px[1] < nose_px[1]
            right_hand_raised = right_wrist_px[1] < nose_px[1]
            
            # Check for waving motion (hands moving horizontally)
            # This would require temporal analysis, for now just check position
            
            # Calculate confidence based on hand positions
            confidence = 0.0
            
            if left_hand_above and right_hand_above:
                confidence += 0.4
            elif left_hand_above or right_hand_above:
                confidence += 0.2
            
            if left_hand_raised and right_hand_raised:
                confidence += 0.4
            elif left_hand_raised or right_hand_raised:
                confidence += 0.2
            
            # Check if hands are spread apart (indicating waving)
            hand_distance = np.sqrt(
                (left_wrist_px[0] - right_wrist_px[0])**2 + 
                (left_wrist_px[1] - right_wrist_px[1])**2
            )
            shoulder_distance = np.sqrt(
                (left_shoulder_px[0] - right_shoulder_px[0])**2 + 
                (left_shoulder_px[1] - right_shoulder_px[1])**2
            )
            
            if hand_distance > shoulder_distance * 1.2:  # Hands spread wider than shoulders
                confidence += 0.2
            
            # Determine if this is an SOS gesture
            is_sos = confidence >= 0.6
            
            return {
                "is_sos": is_sos,
                "confidence": confidence,
                "reason": "sos_detected" if is_sos else "insufficient_gesture",
                "details": {
                    "left_hand_above": left_hand_above,
                    "right_hand_above": right_hand_above,
                    "left_hand_raised": left_hand_raised,
                    "right_hand_raised": right_hand_raised,
                    "hand_distance": hand_distance,
                    "shoulder_distance": shoulder_distance
                }
            }
            
        except Exception as e:
            logger.error(f"Error in SOS gesture detection: {e}")
            return {
                "is_sos": False,
                "confidence": 0.0,
                "reason": "detection_error"
            }
    
    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw pose keypoints on frame
        
        Args:
            frame: Input frame
            keypoints: Pose keypoints
            
        Returns:
            Frame with pose drawn
        """
        if keypoints is None:
            return frame
        
        try:
            # Convert keypoints to MediaPipe format
            landmarks = []
            for kp in keypoints:
                landmark = self.mp_pose.PoseLandmark()
                landmark.x = kp[0]
                landmark.y = kp[1]
                landmark.visibility = kp[2]
                landmarks.append(landmark)
            
            # Create pose landmarks object
            pose_landmarks = self.mp_pose.PoseLandmarks()
            pose_landmarks.landmark.extend(landmarks)
            
            # Draw pose
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing pose: {e}")
            return frame
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "MediaPipe Pose",
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "keypoints_count": 33
        }
