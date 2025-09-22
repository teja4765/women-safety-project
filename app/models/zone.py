"""
Zone model for storing area definitions and risk parameters
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.sql import func
from app.core.database import Base


class Zone(Base):
    """Zone model for defined areas"""
    
    __tablename__ = "zones"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # parking, open_space, park, building
    risk_level = Column(String, nullable=False)  # low, medium, high
    
    # Boundaries (polygon coordinates)
    boundaries = Column(JSON, nullable=False)
    
    # Risk thresholds
    thresholds = Column(JSON, nullable=False)
    
    # Time settings
    quiet_hours = Column(JSON)  # {"start": "22:00", "end": "06:00"}
    
    # Associated cameras
    camera_ids = Column(JSON)  # List of camera IDs
    
    # Statistics
    total_alerts = Column(Integer, default=0)
    last_alert = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Zone(id={self.id}, name={self.name}, type={self.type}, risk_level={self.risk_level})>"
