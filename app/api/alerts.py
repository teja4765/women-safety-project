"""
Alert API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.models.alert import Alert
from app.schemas.alert import AlertResponse, AlertCreate, AlertUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    zone_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),  # Max 1 week
    db: Session = Depends(get_db)
):
    """Get alerts with filtering options"""
    try:
        # Build query
        query = db.query(Alert)
        
        # Apply filters
        if status:
            query = query.filter(Alert.status == status)
        if alert_type:
            query = query.filter(Alert.type == alert_type)
        if camera_id:
            query = query.filter(Alert.camera_id == camera_id)
        if zone_id:
            query = query.filter(Alert.zone_id == zone_id)
        
        # Time filter
        time_threshold = datetime.now() - timedelta(hours=hours)
        query = query.filter(Alert.created_at >= time_threshold)
        
        # Order by creation time (newest first)
        query = query.order_by(Alert.created_at.desc())
        
        # Apply pagination
        alerts = query.offset(skip).limit(limit).all()
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str, db: Session = Depends(get_db)):
    """Get specific alert by ID"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return alert
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    operator_id: str = Query(..., description="ID of the operator acknowledging the alert"),
    feedback: Optional[str] = Query(None, description="Optional feedback from operator"),
    db: Session = Depends(get_db)
):
    """Acknowledge an alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert
        alert.status = "acknowledged"
        alert.acknowledged_by = operator_id
        alert.acknowledged_at = datetime.now()
        if feedback:
            alert.operator_feedback = feedback
        
        db.commit()
        
        return {"message": "Alert acknowledged successfully", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alerts/{alert_id}/escalate")
async def escalate_alert(
    alert_id: str,
    operator_id: str = Query(..., description="ID of the operator escalating the alert"),
    escalation_reason: str = Query(..., description="Reason for escalation"),
    db: Session = Depends(get_db)
):
    """Escalate an alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert
        alert.status = "escalated"
        alert.escalated_by = operator_id
        alert.escalated_at = datetime.now()
        alert.operator_feedback = escalation_reason
        
        db.commit()
        
        return {"message": "Alert escalated successfully", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error escalating alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    operator_id: str = Query(..., description="ID of the operator resolving the alert"),
    false_positive: bool = Query(False, description="Whether this was a false positive"),
    resolution_notes: Optional[str] = Query(None, description="Resolution notes"),
    db: Session = Depends(get_db)
):
    """Resolve an alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert
        alert.status = "resolved"
        alert.acknowledged_by = operator_id
        alert.acknowledged_at = datetime.now()
        alert.false_positive = false_positive
        if resolution_notes:
            alert.operator_feedback = resolution_notes
        
        db.commit()
        
        return {"message": "Alert resolved successfully", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts/stats/summary")
async def get_alert_summary(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """Get alert statistics summary"""
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        # Get basic counts
        total_alerts = db.query(Alert).filter(Alert.created_at >= time_threshold).count()
        
        # Count by status
        status_counts = {}
        for status in ["pending", "acknowledged", "escalated", "resolved"]:
            count = db.query(Alert).filter(
                Alert.created_at >= time_threshold,
                Alert.status == status
            ).count()
            status_counts[status] = count
        
        # Count by type
        type_counts = {}
        for alert_type in ["LONE_WOMAN_NIGHT", "SURROUNDED", "SOS_GESTURE"]:
            count = db.query(Alert).filter(
                Alert.created_at >= time_threshold,
                Alert.type == alert_type
            ).count()
            type_counts[alert_type] = count
        
        # Count false positives
        false_positive_count = db.query(Alert).filter(
            Alert.created_at >= time_threshold,
            Alert.false_positive == True
        ).count()
        
        return {
            "time_window_hours": hours,
            "total_alerts": total_alerts,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "false_positives": false_positive_count,
            "false_positive_rate": false_positive_count / max(total_alerts, 1)
        }
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
