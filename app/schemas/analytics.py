"""
Analytics Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HotspotBase(BaseModel):
    """Base hotspot schema"""
    zone_id: str = Field(..., description="Zone identifier")
    geohash: str = Field(..., description="Geohash for spatial indexing")
    hour_of_week: int = Field(..., ge=0, le=167, description="Hour of week (0-167)")
    date: datetime = Field(..., description="Date of hotspot")
    total_alerts: int = Field(0, description="Total number of alerts")
    lone_woman_alerts: int = Field(0, description="Lone woman alerts")
    surrounded_alerts: int = Field(0, description="Surrounded alerts")
    sos_alerts: int = Field(0, description="SOS gesture alerts")
    risk_score: float = Field(0.0, description="Risk score")
    confidence: float = Field(0.0, description="Confidence score")


class HotspotResponse(HotspotBase):
    """Schema for hotspot responses"""
    id: str = Field(..., description="Hotspot identifier")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")
    
    class Config:
        from_attributes = True


class GenderDistributionBase(BaseModel):
    """Base gender distribution schema"""
    camera_id: str = Field(..., description="Camera identifier")
    zone_id: str = Field(..., description="Zone identifier")
    timestamp: datetime = Field(..., description="Timestamp")
    expected_females: float = Field(0.0, description="Expected number of females")
    expected_males: float = Field(0.0, description="Expected number of males")
    total_people: int = Field(0, description="Total number of people")
    avg_female_confidence: float = Field(0.0, description="Average female confidence")
    avg_male_confidence: float = Field(0.0, description="Average male confidence")
    is_night_time: bool = Field(False, description="Whether it's night time")
    light_level: Optional[float] = Field(None, description="Estimated light level")


class GenderDistributionResponse(GenderDistributionBase):
    """Schema for gender distribution responses"""
    id: str = Field(..., description="Distribution identifier")
    
    class Config:
        from_attributes = True


class SystemMetricsBase(BaseModel):
    """Base system metrics schema"""
    timestamp: datetime = Field(..., description="Timestamp")
    total_cameras: int = Field(0, description="Total number of cameras")
    active_cameras: int = Field(0, description="Number of active cameras")
    processing_fps: float = Field(0.0, description="Processing FPS")
    detection_latency_ms: float = Field(0.0, description="Detection latency in milliseconds")
    alerts_per_hour: float = Field(0.0, description="Alerts per hour")
    false_positive_rate: float = Field(0.0, description="False positive rate")
    operator_response_time_minutes: float = Field(0.0, description="Operator response time in minutes")
    cpu_usage: float = Field(0.0, description="CPU usage percentage")
    memory_usage: float = Field(0.0, description="Memory usage percentage")
    gpu_usage: float = Field(0.0, description="GPU usage percentage")


class SystemMetricsResponse(SystemMetricsBase):
    """Schema for system metrics responses"""
    id: str = Field(..., description="Metrics identifier")
    
    class Config:
        from_attributes = True


class AnalyticsSummary(BaseModel):
    """Schema for analytics summary"""
    time_window_hours: int = Field(..., description="Time window in hours")
    hotspots: dict = Field(..., description="Hotspot summary")
    gender_distribution: dict = Field(..., description="Gender distribution summary")
    system_performance: dict = Field(..., description="System performance summary")
