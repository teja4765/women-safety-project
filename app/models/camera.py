"""
Camera model for storing camera configuration and status
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.sql import func
from app.core.database import Base


class Camera(Base):
    """Camera model for video sources"""
    
    __tablename__ = "cameras"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    zone_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)  # RTSP URL or device path
    
    # Technical specs
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    
    # Position and calibration
    position_x = Column(Float)
    position_y = Column(Float)
    calibration_data = Column(JSON)  # Camera calibration parameters
    
    # Status
    enabled = Column(Boolean, default=True, index=True)
    status = Column(String, default="offline", index=True)  # online, offline, error
    last_heartbeat = Column(DateTime)
    
    # Performance metrics
    processing_fps = Column(Float, default=0.0)
    detection_count = Column(Integer, default=0)
    last_detection = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name={self.name}, zone={self.zone_id}, status={self.status})>"
