"""
Zone API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import logging

from app.core.database import get_db
from app.models.zone import Zone
from app.schemas.zone import ZoneResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/zones", response_model=List[ZoneResponse])
async def get_zones(db: Session = Depends(get_db)):
    """Get all zones"""
    try:
        zones = db.query(Zone).all()
        return zones
        
    except Exception as e:
        logger.error(f"Error getting zones: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/zones/{zone_id}", response_model=ZoneResponse)
async def get_zone(zone_id: str, db: Session = Depends(get_db)):
    """Get specific zone by ID"""
    try:
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")
        
        return zone
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting zone {zone_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
