"""
Health check API endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict
import logging

from app.core.database import get_db
from app.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Safety Detection System",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system status"""
    try:
        # Get system status
        system_status = {
            "status": "healthy",
            "service": "Safety Detection System",
            "version": "1.0.0",
            "components": {
                "api": "healthy",
                "database": "unknown",  # Would check DB connection
                "storage": "unknown",   # Would check storage connection
                "video_processing": "unknown"  # Would check video processor
            }
        }
        
        return system_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/health/cameras")
async def camera_health_check():
    """Camera health check"""
    try:
        # This would get camera status from video processor
        # For now, return mock data
        return {
            "status": "healthy",
            "cameras": {
                "cam_01": {"status": "online", "fps": 5.0},
                "cam_02": {"status": "online", "fps": 5.0},
                "cam_03": {"status": "offline", "fps": 0.0}
            }
        }
        
    except Exception as e:
        logger.error(f"Camera health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
