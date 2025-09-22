"""
Processing modes API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging

from app.core.database import get_db
from app.services.processing_mode_manager import ProcessingModeManager, ProcessingMode
from app.schemas.processing_modes import ProcessingModeResponse, ProcessingModeStats, ModeSwitchRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Global processing mode manager instance
mode_manager = None

async def get_mode_manager():
    """Get processing mode manager instance"""
    global mode_manager
    if mode_manager is None:
        mode_manager = ProcessingModeManager()
        await mode_manager.initialize()
    return mode_manager


@router.get("/processing-modes", response_model=List[ProcessingModeResponse])
async def get_available_processing_modes(db: Session = Depends(get_db)):
    """Get list of available processing modes"""
    try:
        manager = await get_mode_manager()
        
        modes = []
        for mode in ProcessingMode:
            mode_config = manager.get_mode_config(mode)
            performance_settings = manager.get_mode_performance_settings(mode)
            supported_formats = manager.get_supported_formats(mode)
            
            modes.append(ProcessingModeResponse(
                mode=mode.value,
                name=mode_config.get("name", mode.value),
                description=mode_config.get("description", ""),
                fps=performance_settings.get("fps", 5),
                real_time_alerts=performance_settings.get("real_time_alerts", True),
                websocket_updates=performance_settings.get("websocket_updates", True),
                supported_formats=supported_formats,
                temporal_analysis=mode_config.get("temporal_analysis", True),
                performance_monitoring=performance_settings.get("performance_monitoring", True)
            ))
        
        return modes
        
    except Exception as e:
        logger.error(f"Error getting processing modes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/active", response_model=Dict[str, str])
async def get_active_processing_modes(db: Session = Depends(get_db)):
    """Get active processing modes for all cameras"""
    try:
        manager = await get_mode_manager()
        active_modes = manager.get_all_active_modes()
        
        return {
            camera_id: mode.value 
            for camera_id, mode in active_modes.items()
        }
        
    except Exception as e:
        logger.error(f"Error getting active processing modes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/camera/{camera_id}", response_model=ProcessingModeResponse)
async def get_camera_processing_mode(camera_id: str, db: Session = Depends(get_db)):
    """Get processing mode for a specific camera"""
    try:
        manager = await get_mode_manager()
        active_mode = manager.get_active_mode(camera_id)
        
        if not active_mode:
            raise HTTPException(status_code=404, detail="No active processing mode found for camera")
        
        mode_config = manager.get_mode_config(active_mode)
        performance_settings = manager.get_mode_performance_settings(active_mode)
        supported_formats = manager.get_supported_formats(active_mode)
        
        return ProcessingModeResponse(
            mode=active_mode.value,
            name=mode_config.get("name", active_mode.value),
            description=mode_config.get("description", ""),
            fps=performance_settings.get("fps", 5),
            real_time_alerts=performance_settings.get("real_time_alerts", True),
            websocket_updates=performance_settings.get("websocket_updates", True),
            supported_formats=supported_formats,
            temporal_analysis=mode_config.get("temporal_analysis", True),
            performance_monitoring=performance_settings.get("performance_monitoring", True)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera processing mode {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/processing-modes/camera/{camera_id}/switch")
async def switch_camera_processing_mode(
    camera_id: str,
    request: ModeSwitchRequest,
    db: Session = Depends(get_db)
):
    """Switch processing mode for a specific camera"""
    try:
        manager = await get_mode_manager()
        
        # Validate new mode
        try:
            new_mode = ProcessingMode(request.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid processing mode: {request.mode}")
        
        # Check if mode can be switched
        if not manager.can_switch_mode(camera_id, new_mode):
            raise HTTPException(
                status_code=400, 
                detail="Mode switch not allowed (cooldown active or switching disabled)"
            )
        
        # Perform the switch
        success = manager.switch_mode(camera_id, new_mode)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to switch processing mode")
        
        logger.info(f"Switched camera {camera_id} to processing mode {request.mode}")
        
        return {
            "message": f"Camera {camera_id} switched to {request.mode} mode",
            "camera_id": camera_id,
            "new_mode": request.mode,
            "timestamp": manager.mode_switches.get(camera_id).isoformat() if camera_id in manager.mode_switches else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching camera processing mode {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/validate-input")
async def validate_input_for_mode(
    input_data: str = Query(..., description="Input data to validate"),
    mode: str = Query(..., description="Processing mode to validate against"),
    db: Session = Depends(get_db)
):
    """Validate input data for a specific processing mode"""
    try:
        manager = await get_mode_manager()
        
        # Validate mode
        try:
            processing_mode = ProcessingMode(mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid processing mode: {mode}")
        
        # Validate input
        is_valid = manager.validate_input_for_mode(input_data, processing_mode)
        
        return {
            "input_data": input_data,
            "mode": mode,
            "is_valid": is_valid,
            "supported_formats": manager.get_supported_formats(processing_mode) if not is_valid else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating input for mode: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/auto-select")
async def auto_select_processing_mode(
    input_type: str = Query(..., description="Type of input (rtsp_url, file_upload, file_path)"),
    manual_mode: Optional[str] = Query(None, description="Manual mode override"),
    db: Session = Depends(get_db)
):
    """Auto-select appropriate processing mode based on input type"""
    try:
        manager = await get_mode_manager()
        
        selected_mode = manager.select_processing_mode(input_type, manual_mode)
        
        return {
            "input_type": input_type,
            "selected_mode": selected_mode.value,
            "mode_description": manager.get_mode_description(selected_mode),
            "performance_settings": manager.get_mode_performance_settings(selected_mode),
            "supported_formats": manager.get_supported_formats(selected_mode),
            "manual_override_used": manual_mode is not None
        }
        
    except Exception as e:
        logger.error(f"Error auto-selecting processing mode: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/statistics", response_model=ProcessingModeStats)
async def get_processing_mode_statistics(db: Session = Depends(get_db)):
    """Get statistics about processing modes"""
    try:
        manager = await get_mode_manager()
        stats = manager.get_mode_statistics()
        
        return ProcessingModeStats(
            total_cameras=stats.get("total_cameras", 0),
            mode_distribution=stats.get("mode_distribution", {}),
            available_modes=stats.get("available_modes", []),
            last_updated=stats.get("last_updated", "")
        )
        
    except Exception as e:
        logger.error(f"Error getting processing mode statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/switch-history")
async def get_mode_switch_history(db: Session = Depends(get_db)):
    """Get history of processing mode switches"""
    try:
        manager = await get_mode_manager()
        history = manager.get_mode_switch_history()
        
        return {
            "switch_history": history,
            "total_switches": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting mode switch history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/processing-modes/reset-switches")
async def reset_mode_switches(db: Session = Depends(get_db)):
    """Reset mode switch tracking (useful for testing)"""
    try:
        manager = await get_mode_manager()
        manager.reset_mode_switches()
        
        return {"message": "Mode switch tracking reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting mode switches: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing-modes/configuration")
async def get_processing_mode_configuration(db: Session = Depends(get_db)):
    """Get full processing mode configuration"""
    try:
        manager = await get_mode_manager()
        
        return {
            "global_settings": manager.get_global_config(),
            "mode_selection_rules": manager.get_mode_selection_rules(),
            "available_modes": {
                mode.value: {
                    "config": manager.get_mode_config(mode),
                    "performance_settings": manager.get_mode_performance_settings(mode),
                    "risk_detection_settings": manager.get_mode_risk_detection_settings(mode),
                    "privacy_settings": manager.get_mode_privacy_settings(mode),
                    "supported_formats": manager.get_supported_formats(mode)
                }
                for mode in ProcessingMode
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting processing mode configuration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
