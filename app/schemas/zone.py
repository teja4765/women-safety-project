"""
Zone Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ZoneBase(BaseModel):
    """Base zone schema"""
    name: str = Field(..., description="Zone name")
    type: str = Field(..., description="Zone type")
    risk_level: str = Field(..., description="Risk level")
    boundaries: List[List[float]] = Field(..., description="Zone boundaries")
    thresholds: Dict[str, Any] = Field(..., description="Risk thresholds")
    quiet_hours: Optional[Dict[str, str]] = Field(None, description="Quiet hours")
    camera_ids: Optional[List[str]] = Field(None, description="Associated camera IDs")


class ZoneCreate(ZoneBase):
    """Schema for creating zones"""
    pass


class ZoneUpdate(BaseModel):
    """Schema for updating zones"""
    name: Optional[str] = Field(None, description="Zone name")
    type: Optional[str] = Field(None, description="Zone type")
    risk_level: Optional[str] = Field(None, description="Risk level")
    boundaries: Optional[List[List[float]]] = Field(None, description="Zone boundaries")
    thresholds: Optional[Dict[str, Any]] = Field(None, description="Risk thresholds")
    quiet_hours: Optional[Dict[str, str]] = Field(None, description="Quiet hours")
    camera_ids: Optional[List[str]] = Field(None, description="Associated camera IDs")


class ZoneResponse(ZoneBase):
    """Schema for zone responses"""
    id: str = Field(..., description="Zone identifier")
    total_alerts: int = Field(0, description="Total number of alerts")
    last_alert: Optional[datetime] = Field(None, description="Last alert time")
    created_at: datetime = Field(..., description="Zone creation time")
    updated_at: datetime = Field(..., description="Last update time")
    
    class Config:
        from_attributes = True
