"""
Analytics models for storing aggregated data and hotspots
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Index, Boolean
from sqlalchemy.sql import func
from app.core.database import Base


class Hotspot(Base):
    """Hotspot model for risk area analytics"""
    
    __tablename__ = "hotspots"
    
    id = Column(String, primary_key=True, index=True)
    zone_id = Column(String, nullable=False, index=True)
    geohash = Column(String, nullable=False, index=True)  # For spatial indexing
    
    # Time period
    hour_of_week = Column(Integer, nullable=False, index=True)  # 0-167 (24*7)
    date = Column(DateTime, nullable=False, index=True)
    
    # Alert statistics
    total_alerts = Column(Integer, default=0)
    lone_woman_alerts = Column(Integer, default=0)
    surrounded_alerts = Column(Integer, default=0)
    sos_alerts = Column(Integer, default=0)
    
    # Risk metrics
    risk_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_zone_hour_date', 'zone_id', 'hour_of_week', 'date'),
    )
    
    def __repr__(self):
        return f"<Hotspot(zone={self.zone_id}, hour={self.hour_of_week}, alerts={self.total_alerts})>"


class GenderDistribution(Base):
    """Gender distribution model for real-time analytics"""
    
    __tablename__ = "gender_distributions"
    
    id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, nullable=False, index=True)
    zone_id = Column(String, nullable=False, index=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Counts (expected values based on confidence)
    expected_females = Column(Float, default=0.0)
    expected_males = Column(Float, default=0.0)
    total_people = Column(Integer, default=0)
    
    # Confidence metrics
    avg_female_confidence = Column(Float, default=0.0)
    avg_male_confidence = Column(Float, default=0.0)
    
    # Context
    is_night_time = Column(Boolean, default=False)
    light_level = Column(Float)  # Estimated light level
    
    def __repr__(self):
        return f"<GenderDistribution(camera={self.camera_id}, females={self.expected_females}, males={self.expected_males})>"


class SystemMetrics(Base):
    """System performance metrics"""
    
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Performance metrics
    total_cameras = Column(Integer, default=0)
    active_cameras = Column(Integer, default=0)
    processing_fps = Column(Float, default=0.0)
    detection_latency_ms = Column(Float, default=0.0)
    
    # Alert metrics
    alerts_per_hour = Column(Float, default=0.0)
    false_positive_rate = Column(Float, default=0.0)
    operator_response_time_minutes = Column(Float, default=0.0)
    
    # System health
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    gpu_usage = Column(Float, default=0.0)
    
    def __repr__(self):
        return f"<SystemMetrics(timestamp={self.timestamp}, fps={self.processing_fps})>"
