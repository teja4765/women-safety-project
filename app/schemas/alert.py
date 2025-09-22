"""
Alert Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class AlertBase(BaseModel):
    """Base alert schema"""
    type: str = Field(..., description="Alert type")
    camera_id: str = Field(..., description="Camera identifier")
    zone_id: str = Field(..., description="Zone identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    severity: float = Field(..., ge=0.0, le=1.0, description="Risk severity")
    description: str = Field(..., description="Alert description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AlertCreate(AlertBase):
    """Schema for creating alerts"""
    start_time: datetime = Field(..., description="Alert start time")
    end_time: datetime = Field(..., description="Alert end time")


class AlertUpdate(BaseModel):
    """Schema for updating alerts"""
    status: Optional[str] = Field(None, description="Alert status")
    acknowledged_by: Optional[str] = Field(None, description="Operator who acknowledged")
    escalated_by: Optional[str] = Field(None, description="Operator who escalated")
    operator_feedback: Optional[str] = Field(None, description="Operator feedback")
    false_positive: Optional[bool] = Field(None, description="Whether this was a false positive")


class AlertResponse(AlertBase):
    """Schema for alert responses"""
    id: str = Field(..., description="Alert identifier")
    start_time: datetime = Field(..., description="Alert start time")
    end_time: datetime = Field(..., description="Alert end time")
    created_at: datetime = Field(..., description="Alert creation time")
    status: str = Field(..., description="Alert status")
    clip_url: Optional[str] = Field(None, description="URL to event video clip")
    thumbnail_url: Optional[str] = Field(None, description="URL to event thumbnail")
    acknowledged_by: Optional[str] = Field(None, description="Operator who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
    escalated_by: Optional[str] = Field(None, description="Operator who escalated")
    escalated_at: Optional[datetime] = Field(None, description="Escalation time")
    false_positive: Optional[bool] = Field(None, description="Whether this was a false positive")
    
    class Config:
        from_attributes = True


class AlertSummary(BaseModel):
    """Schema for alert summary statistics"""
    time_window_hours: int = Field(..., description="Time window in hours")
    total_alerts: int = Field(..., description="Total number of alerts")
    status_breakdown: Dict[str, int] = Field(..., description="Breakdown by status")
    type_breakdown: Dict[str, int] = Field(..., description="Breakdown by type")
    false_positives: int = Field(..., description="Number of false positives")
    false_positive_rate: float = Field(..., description="False positive rate")
