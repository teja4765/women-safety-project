"""
Alert model for storing safety detection events
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class Alert(Base):
    """Alert model for safety events"""
    
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True, index=True)
    type = Column(String, nullable=False, index=True)  # LONE_WOMAN, SURROUNDED, SOS_GESTURE
    camera_id = Column(String, nullable=False, index=True)
    zone_id = Column(String, nullable=False, index=True)
    
    # Timestamps
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Confidence and severity
    confidence = Column(Float, nullable=False)
    severity = Column(Float, nullable=False)
    
    # Event details
    description = Column(Text)
    # 'metadata' attribute name is reserved by SQLAlchemy's Declarative API.
    # Map to DB column named 'metadata' but expose as 'meta' attribute.
    meta = Column('metadata', JSON)
    
    # Media
    clip_url = Column(String)
    thumbnail_url = Column(String)
    
    # Status
    status = Column(String, default="pending", index=True)  # pending, acknowledged, escalated, resolved
    acknowledged_by = Column(String)
    acknowledged_at = Column(DateTime)
    escalated_by = Column(String)
    escalated_at = Column(DateTime)
    
    # Feedback
    operator_feedback = Column(Text)
    false_positive = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Alert(id={self.id}, type={self.type}, camera={self.camera_id}, confidence={self.confidence})>"

    # Avoid defining any attribute named 'metadata' to prevent SQLAlchemy conflicts
