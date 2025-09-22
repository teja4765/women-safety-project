"""
Video analysis Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoAnalysisJob(BaseModel):
    """Schema for video analysis job creation"""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    zone_id: str = Field(..., description="Zone ID for analysis context")
    camera_id: str = Field(..., description="Camera ID")
    processing_mode: str = Field(..., description="Processing mode: batch or realtime")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress percentage")
    created_at: datetime = Field(..., description="Job creation time")


class VideoAnalysisResponse(BaseModel):
    """Schema for video analysis job response"""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    zone_id: str = Field(..., description="Zone ID for analysis context")
    camera_id: str = Field(..., description="Camera ID")
    processing_mode: str = Field(..., description="Processing mode: batch or realtime")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress percentage")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    total_frames: int = Field(0, description="Total frames in video")
    processed_frames: int = Field(0, description="Frames processed so far")
    alerts_found: int = Field(0, description="Number of alerts found")
    errors: List[str] = Field(default_factory=list, description="Processing errors")


class VideoAnalysisStatus(BaseModel):
    """Schema for video analysis system status"""
    status: str = Field(..., description="System status")
    active_jobs: int = Field(..., description="Number of active jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    total_jobs: int = Field(..., description="Total number of jobs")
    processor_initialized: bool = Field(..., description="Whether processor is initialized")


class VideoAnalysisResults(BaseModel):
    """Schema for video analysis results"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    processing_report: Dict[str, Any] = Field(..., description="Processing report")
    alerts: List[Dict[str, Any]] = Field(..., description="Detected alerts")
    sample_frames: List[str] = Field(..., description="Sample annotated frames")
    output_directory: str = Field(..., description="Output directory path")
    total_alerts: int = Field(..., description="Total number of alerts")
    total_frames: int = Field(..., description="Total number of frames")


class VideoUploadRequest(BaseModel):
    """Schema for video upload request"""
    zone_id: str = Field(..., description="Zone ID for analysis context")
    camera_id: Optional[str] = Field(None, description="Camera ID (optional)")
    processing_mode: str = Field("batch", description="Processing mode: batch or realtime")


class VideoProcessingRequest(BaseModel):
    """Schema for video processing request"""
    video_path: str = Field(..., description="Path to video file")
    zone_id: str = Field(..., description="Zone ID for analysis context")
    camera_id: Optional[str] = Field(None, description="Camera ID (optional)")
    processing_mode: str = Field("batch", description="Processing mode: batch or realtime")
