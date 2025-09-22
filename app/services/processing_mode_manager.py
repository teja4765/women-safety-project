"""
Processing mode manager for handling different processing modes
"""

import asyncio
import yaml
import logging
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode enumeration"""
    LIVE_CCTV = "live_cctv"
    VIDEO_FILE_BATCH = "video_file_batch"
    VIDEO_FILE_REALTIME = "video_file_realtime"


class ProcessingModeManager:
    """Manages different processing modes and their configurations"""
    
    def __init__(self):
        self.modes_config = {}
        self.active_modes = {}
        self.mode_switches = {}  # Track mode switches for cooldown
        
    async def initialize(self):
        """Initialize the processing mode manager"""
        try:
            # Load processing modes configuration
            with open("config/processing_modes.yaml", "r") as f:
                self.modes_config = yaml.safe_load(f)
            
            logger.info("Processing mode manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processing mode manager: {e}")
            raise
    
    def get_mode_config(self, mode: ProcessingMode) -> Dict:
        """Get configuration for a specific processing mode"""
        try:
            mode_key = mode.value
            return self.modes_config.get("processing_modes", {}).get(mode_key, {})
        except Exception as e:
            logger.error(f"Error getting mode config for {mode}: {e}")
            return {}
    
    def get_global_config(self) -> Dict:
        """Get global processing configuration"""
        return self.modes_config.get("global_settings", {})
    
    def get_mode_selection_rules(self) -> Dict:
        """Get mode selection rules"""
        return self.modes_config.get("mode_selection", {})
    
    def select_processing_mode(self, input_type: str, manual_mode: Optional[str] = None) -> ProcessingMode:
        """
        Select appropriate processing mode based on input type and manual override
        
        Args:
            input_type: Type of input ("rtsp_url", "file_upload", "file_path")
            manual_mode: Manual mode override if provided
            
        Returns:
            Selected processing mode
        """
        try:
            # Check for manual override
            if manual_mode:
                if self._is_valid_mode(manual_mode):
                    logger.info(f"Using manual mode override: {manual_mode}")
                    return ProcessingMode(manual_mode)
                else:
                    logger.warning(f"Invalid manual mode: {manual_mode}, using auto-selection")
            
            # Auto-select based on input type
            auto_select_rules = self.get_mode_selection_rules().get("auto_select", {})
            selected_mode = auto_select_rules.get(input_type, "video_file_batch")
            
            if self._is_valid_mode(selected_mode):
                logger.info(f"Auto-selected processing mode: {selected_mode} for input type: {input_type}")
                return ProcessingMode(selected_mode)
            else:
                logger.warning(f"Invalid auto-selected mode: {selected_mode}, using default")
                return ProcessingMode.VIDEO_FILE_BATCH
                
        except Exception as e:
            logger.error(f"Error selecting processing mode: {e}")
            return ProcessingMode.VIDEO_FILE_BATCH
    
    def _is_valid_mode(self, mode: str) -> bool:
        """Check if a mode is valid"""
        try:
            ProcessingMode(mode)
            return True
        except ValueError:
            return False
    
    def can_switch_mode(self, camera_id: str, new_mode: ProcessingMode) -> bool:
        """
        Check if mode can be switched for a specific camera
        
        Args:
            camera_id: Camera identifier
            new_mode: New processing mode
            
        Returns:
            True if mode can be switched
        """
        try:
            mode_switching_config = self.get_mode_selection_rules().get("mode_switching", {})
            
            if not mode_switching_config.get("allow_runtime_switch", True):
                return False
            
            # Check cooldown period
            cooldown_seconds = mode_switching_config.get("switch_cooldown_seconds", 60)
            last_switch = self.mode_switches.get(camera_id)
            
            if last_switch:
                time_since_switch = (datetime.now() - last_switch).total_seconds()
                if time_since_switch < cooldown_seconds:
                    logger.warning(f"Mode switch cooldown active for camera {camera_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking mode switch capability: {e}")
            return False
    
    def switch_mode(self, camera_id: str, new_mode: ProcessingMode) -> bool:
        """
        Switch processing mode for a specific camera
        
        Args:
            camera_id: Camera identifier
            new_mode: New processing mode
            
        Returns:
            True if mode was switched successfully
        """
        try:
            if not self.can_switch_mode(camera_id, new_mode):
                return False
            
            # Record the mode switch
            self.mode_switches[camera_id] = datetime.now()
            self.active_modes[camera_id] = new_mode
            
            logger.info(f"Switched camera {camera_id} to mode {new_mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching mode for camera {camera_id}: {e}")
            return False
    
    def get_active_mode(self, camera_id: str) -> Optional[ProcessingMode]:
        """Get active processing mode for a camera"""
        return self.active_modes.get(camera_id)
    
    def get_all_active_modes(self) -> Dict[str, ProcessingMode]:
        """Get all active processing modes"""
        return self.active_modes.copy()
    
    def get_mode_performance_settings(self, mode: ProcessingMode) -> Dict:
        """Get performance settings for a specific mode"""
        try:
            mode_config = self.get_mode_config(mode)
            global_config = self.get_global_config()
            
            return {
                "fps": mode_config.get("fps", 5),
                "frame_skip": mode_config.get("frame_skip", 6),
                "buffer_size": mode_config.get("buffer_size", 30),
                "temporal_analysis": mode_config.get("temporal_analysis", True),
                "real_time_alerts": mode_config.get("real_time_alerts", True),
                "performance_monitoring": mode_config.get("performance_monitoring", True),
                "websocket_updates": mode_config.get("websocket_updates", True),
                "max_concurrent_cameras": global_config.get("performance", {}).get("max_concurrent_cameras", 8),
                "gpu_memory_limit_mb": global_config.get("performance", {}).get("gpu_memory_limit_mb", 2048),
                "cpu_usage_limit": global_config.get("performance", {}).get("cpu_usage_limit", 0.8)
            }
        except Exception as e:
            logger.error(f"Error getting performance settings for mode {mode}: {e}")
            return {}
    
    def get_mode_risk_detection_settings(self, mode: ProcessingMode) -> Dict:
        """Get risk detection settings for a specific mode"""
        try:
            mode_config = self.get_mode_config(mode)
            return mode_config.get("risk_detection", {})
        except Exception as e:
            logger.error(f"Error getting risk detection settings for mode {mode}: {e}")
            return {}
    
    def get_mode_privacy_settings(self, mode: ProcessingMode) -> Dict:
        """Get privacy settings for a specific mode"""
        try:
            global_config = self.get_global_config()
            return global_config.get("privacy", {})
        except Exception as e:
            logger.error(f"Error getting privacy settings for mode {mode}: {e}")
            return {}
    
    def get_supported_formats(self, mode: ProcessingMode) -> List[str]:
        """Get supported file formats for a specific mode"""
        try:
            if mode == ProcessingMode.LIVE_CCTV:
                return ["rtsp", "http", "udp"]  # Live stream formats
            else:
                mode_config = self.get_mode_config(mode)
                return mode_config.get("batch_settings", {}).get("supported_formats", ["mp4", "avi", "mov"])
        except Exception as e:
            logger.error(f"Error getting supported formats for mode {mode}: {e}")
            return ["mp4"]
    
    def validate_input_for_mode(self, input_data: str, mode: ProcessingMode) -> bool:
        """
        Validate input data for a specific processing mode
        
        Args:
            input_data: Input data (URL, file path, etc.)
            mode: Processing mode
            
        Returns:
            True if input is valid for the mode
        """
        try:
            if mode == ProcessingMode.LIVE_CCTV:
                # Check if it's a valid live stream URL
                return (input_data.startswith("rtsp://") or 
                       input_data.startswith("http://") or 
                       input_data.startswith("udp://") or
                       input_data.isdigit())  # Webcam index
            
            else:
                # Check if it's a valid file path and format
                import os
                if not os.path.exists(input_data):
                    return False
                
                file_extension = os.path.splitext(input_data)[1].lower().lstrip('.')
                supported_formats = self.get_supported_formats(mode)
                return file_extension in supported_formats
                
        except Exception as e:
            logger.error(f"Error validating input for mode {mode}: {e}")
            return False
    
    def get_mode_description(self, mode: ProcessingMode) -> str:
        """Get human-readable description of a processing mode"""
        try:
            mode_config = self.get_mode_config(mode)
            return mode_config.get("description", f"Processing mode: {mode.value}")
        except Exception as e:
            logger.error(f"Error getting mode description for {mode}: {e}")
            return f"Processing mode: {mode.value}"
    
    def get_mode_statistics(self) -> Dict:
        """Get statistics about processing modes"""
        try:
            total_cameras = len(self.active_modes)
            mode_counts = {}
            
            for mode in self.active_modes.values():
                mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1
            
            return {
                "total_cameras": total_cameras,
                "mode_distribution": mode_counts,
                "available_modes": [mode.value for mode in ProcessingMode],
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting mode statistics: {e}")
            return {}
    
    def reset_mode_switches(self):
        """Reset mode switch tracking (useful for testing)"""
        self.mode_switches.clear()
        logger.info("Mode switch tracking reset")
    
    def get_mode_switch_history(self) -> Dict:
        """Get history of mode switches"""
        return {
            camera_id: switch_time.isoformat() 
            for camera_id, switch_time in self.mode_switches.items()
        }
