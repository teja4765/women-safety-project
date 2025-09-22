"""
Analytics API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.models.analytics import Hotspot, GenderDistribution, SystemMetrics
from app.schemas.analytics import HotspotResponse, GenderDistributionResponse, SystemMetricsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/analytics/hotspots", response_model=List[HotspotResponse])
async def get_hotspots(
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
):
    """Get hotspot analytics"""
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        query = db.query(Hotspot).filter(Hotspot.date >= time_threshold)
        
        if zone_id:
            query = query.filter(Hotspot.zone_id == zone_id)
        
        hotspots = query.order_by(Hotspot.risk_score.desc()).all()
        return hotspots
        
    except Exception as e:
        logger.error(f"Error getting hotspots: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/gender-distribution", response_model=List[GenderDistributionResponse])
async def get_gender_distribution(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    zone_id: Optional[str] = Query(None, description="Filter by zone ID"),
    hours: int = Query(1, ge=1, le=24, description="Time window in hours"),
    db: Session = Depends(get_db)
):
    """Get gender distribution analytics"""
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        query = db.query(GenderDistribution).filter(GenderDistribution.timestamp >= time_threshold)
        
        if camera_id:
            query = query.filter(GenderDistribution.camera_id == camera_id)
        if zone_id:
            query = query.filter(GenderDistribution.zone_id == zone_id)
        
        distributions = query.order_by(GenderDistribution.timestamp.desc()).all()
        return distributions
        
    except Exception as e:
        logger.error(f"Error getting gender distribution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/system-metrics", response_model=List[SystemMetricsResponse])
async def get_system_metrics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
):
    """Get system performance metrics"""
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        metrics = db.query(SystemMetrics).filter(
            SystemMetrics.timestamp >= time_threshold
        ).order_by(SystemMetrics.timestamp.desc()).all()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/summary")
async def get_analytics_summary(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
):
    """Get analytics summary"""
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        # Get hotspot summary
        total_hotspots = db.query(Hotspot).filter(Hotspot.date >= time_threshold).count()
        high_risk_hotspots = db.query(Hotspot).filter(
            Hotspot.date >= time_threshold,
            Hotspot.risk_score >= 0.7
        ).count()
        
        # Get gender distribution summary
        avg_females = db.query(GenderDistribution).filter(
            GenderDistribution.timestamp >= time_threshold
        ).with_entities(
            db.func.avg(GenderDistribution.expected_females)
        ).scalar() or 0.0
        
        avg_males = db.query(GenderDistribution).filter(
            GenderDistribution.timestamp >= time_threshold
        ).with_entities(
            db.func.avg(GenderDistribution.expected_males)
        ).scalar() or 0.0
        
        # Get system metrics summary
        latest_metrics = db.query(SystemMetrics).order_by(
            SystemMetrics.timestamp.desc()
        ).first()
        
        return {
            "time_window_hours": hours,
            "hotspots": {
                "total": total_hotspots,
                "high_risk": high_risk_hotspots
            },
            "gender_distribution": {
                "avg_females": round(avg_females, 2),
                "avg_males": round(avg_males, 2)
            },
            "system_performance": {
                "total_cameras": latest_metrics.total_cameras if latest_metrics else 0,
                "active_cameras": latest_metrics.active_cameras if latest_metrics else 0,
                "processing_fps": latest_metrics.processing_fps if latest_metrics else 0.0,
                "alerts_per_hour": latest_metrics.alerts_per_hour if latest_metrics else 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
