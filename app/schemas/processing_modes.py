"""
Processing modes Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class ProcessingModeResponse(BaseModel):
    """Schema for processing mode response"""
    mode: str = Field(..., description="Processing mode identifier")
    name: str = Field(..., description="Human-readable mode name")
    description: str = Field(..., description="Mode description")
    fps: int = Field(..., description="Target processing FPS")
    real_time_alerts: bool = Field(..., description="Whether real-time alerts are enabled")
    websocket_updates: bool = Field(..., description="Whether WebSocket updates are enabled")
    supported_formats: List[str] = Field(..., description="Supported input formats")
    temporal_analysis: bool = Field(..., description="Whether temporal analysis is enabled")
    performance_monitoring: bool = Field(..., description="Whether performance monitoring is enabled")


class ModeSwitchRequest(BaseModel):
    """Schema for mode switch request"""
    mode: str = Field(..., description="New processing mode")
    reason: Optional[str] = Field(None, description="Reason for mode switch")


class ProcessingModeStats(BaseModel):
    """Schema for processing mode statistics"""
    total_cameras: int = Field(..., description="Total number of cameras")
    mode_distribution: Dict[str, int] = Field(..., description="Distribution of cameras by mode")
    available_modes: List[str] = Field(..., description="List of available modes")
    last_updated: str = Field(..., description="Last update timestamp")


class ProcessingModeConfiguration(BaseModel):
    """Schema for processing mode configuration"""
    global_settings: Dict = Field(..., description="Global processing settings")
    mode_selection_rules: Dict = Field(..., description="Mode selection rules")
    available_modes: Dict[str, Dict] = Field(..., description="Available modes configuration")


class ModeValidationResponse(BaseModel):
    """Schema for mode validation response"""
    input_data: str = Field(..., description="Input data that was validated")
    mode: str = Field(..., description="Processing mode that was validated")
    is_valid: bool = Field(..., description="Whether input is valid for the mode")
    supported_formats: Optional[List[str]] = Field(None, description="Supported formats if invalid")


class AutoSelectResponse(BaseModel):
    """Schema for auto-select response"""
    input_type: str = Field(..., description="Type of input")
    selected_mode: str = Field(..., description="Selected processing mode")
    mode_description: str = Field(..., description="Description of selected mode")
    performance_settings: Dict = Field(..., description="Performance settings for the mode")
    supported_formats: List[str] = Field(..., description="Supported formats for the mode")
    manual_override_used: bool = Field(..., description="Whether manual override was used")


class ModeSwitchResponse(BaseModel):
    """Schema for mode switch response"""
    message: str = Field(..., description="Success message")
    camera_id: str = Field(..., description="Camera identifier")
    new_mode: str = Field(..., description="New processing mode")
    timestamp: Optional[str] = Field(None, description="Switch timestamp")


class ModeSwitchHistory(BaseModel):
    """Schema for mode switch history"""
    switch_history: Dict[str, str] = Field(..., description="History of mode switches")
    total_switches: int = Field(..., description="Total number of switches")
