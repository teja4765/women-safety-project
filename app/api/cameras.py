"""
Camera API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.core.database import get_db
from app.models.camera import Camera
from app.schemas.camera import CameraResponse, CameraStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/cameras", response_model=List[CameraResponse])
async def get_cameras(
    enabled_only: bool = Query(False, description="Return only enabled cameras"),
    db: Session = Depends(get_db)
):
    """Get all cameras"""
    try:
        query = db.query(Camera)
        
        if enabled_only:
            query = query.filter(Camera.enabled == True)
        
        cameras = query.all()
        return cameras
        
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """Get specific camera by ID"""
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        return camera
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cameras/{camera_id}/status", response_model=CameraStatus)
async def get_camera_status(camera_id: str):
    """Get camera status and performance metrics"""
    try:
        # This would get real-time status from video processor
        # For now, return mock data
        return {
            "camera_id": camera_id,
            "status": "online",
            "processing_fps": 5.0,
            "detection_count": 42,
            "last_detection": "2025-01-04T10:30:00Z",
            "uptime_seconds": 3600
        }
        
    except Exception as e:
        logger.error(f"Error getting camera status {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cameras/{camera_id}/enable")
async def enable_camera(camera_id: str, db: Session = Depends(get_db)):
    """Enable a camera"""
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        camera.enabled = True
        db.commit()
        
        return {"message": f"Camera {camera_id} enabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cameras/{camera_id}/disable")
async def disable_camera(camera_id: str, db: Session = Depends(get_db)):
    """Disable a camera"""
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        camera.enabled = False
        db.commit()
        
        return {"message": f"Camera {camera_id} disabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cameras/stats/summary")
async def get_camera_summary(db: Session = Depends(get_db)):
    """Get camera statistics summary"""
    try:
        total_cameras = db.query(Camera).count()
        enabled_cameras = db.query(Camera).filter(Camera.enabled == True).count()
        online_cameras = db.query(Camera).filter(Camera.status == "online").count()
        
        return {
            "total_cameras": total_cameras,
            "enabled_cameras": enabled_cameras,
            "online_cameras": online_cameras,
            "offline_cameras": total_cameras - online_cameras
        }
        
    except Exception as e:
        logger.error(f"Error getting camera summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
