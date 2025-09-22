"""
Camera Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class CameraBase(BaseModel):
    """Base camera schema"""
    name: str = Field(..., description="Camera name")
    zone_id: str = Field(..., description="Zone identifier")
    source: str = Field(..., description="Video source URL or path")
    resolution_width: int = Field(1920, description="Video width")
    resolution_height: int = Field(1080, description="Video height")
    fps: int = Field(30, description="Frames per second")
    position_x: Optional[float] = Field(None, description="X position coordinate")
    position_y: Optional[float] = Field(None, description="Y position coordinate")
    calibration_data: Optional[Dict[str, Any]] = Field(None, description="Camera calibration data")


class CameraCreate(CameraBase):
    """Schema for creating cameras"""
    pass


class CameraUpdate(BaseModel):
    """Schema for updating cameras"""
    name: Optional[str] = Field(None, description="Camera name")
    enabled: Optional[bool] = Field(None, description="Whether camera is enabled")
    resolution_width: Optional[int] = Field(None, description="Video width")
    resolution_height: Optional[int] = Field(None, description="Video height")
    fps: Optional[int] = Field(None, description="Frames per second")
    position_x: Optional[float] = Field(None, description="X position coordinate")
    position_y: Optional[float] = Field(None, description="Y position coordinate")
    calibration_data: Optional[Dict[str, Any]] = Field(None, description="Camera calibration data")


class CameraResponse(CameraBase):
    """Schema for camera responses"""
    id: str = Field(..., description="Camera identifier")
    enabled: bool = Field(..., description="Whether camera is enabled")
    status: str = Field(..., description="Camera status")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat time")
    processing_fps: float = Field(0.0, description="Current processing FPS")
    detection_count: int = Field(0, description="Total detection count")
    last_detection: Optional[datetime] = Field(None, description="Last detection time")
    created_at: datetime = Field(..., description="Camera creation time")
    updated_at: datetime = Field(..., description="Last update time")
    
    class Config:
        from_attributes = True


class CameraStatus(BaseModel):
    """Schema for camera status"""
    camera_id: str = Field(..., description="Camera identifier")
    status: str = Field(..., description="Camera status")
    processing_fps: float = Field(..., description="Current processing FPS")
    detection_count: int = Field(..., description="Total detection count")
    last_detection: Optional[datetime] = Field(None, description="Last detection time")
    uptime_seconds: int = Field(..., description="Camera uptime in seconds")


class CameraSummary(BaseModel):
    """Schema for camera summary statistics"""
    total_cameras: int = Field(..., description="Total number of cameras")
    enabled_cameras: int = Field(..., description="Number of enabled cameras")
    online_cameras: int = Field(..., description="Number of online cameras")
    offline_cameras: int = Field(..., description="Number of offline cameras")
