"""
Cleanup background tasks
"""

from celery import shared_task
from datetime import datetime, timedelta
import logging

from app.core.database import SessionLocal
from app.models.alert import Alert
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


@shared_task
def cleanup_old_alerts():
    """Clean up old alerts and associated files"""
    try:
        db = SessionLocal()
        
        # Find alerts older than retention period
        cutoff_date = datetime.now() - timedelta(days=30)
        old_alerts = db.query(Alert).filter(
            Alert.created_at < cutoff_date
        ).all()
        
        storage_service = StorageService()
        
        deleted_count = 0
        for alert in old_alerts:
            try:
                # Delete associated files
                if alert.clip_url:
                    # Extract object name from URL
                    object_name = alert.clip_url.split('/')[-1]
                    storage_service.delete_file(f"alerts/{alert.id}/{object_name}")
                
                if alert.thumbnail_url:
                    object_name = alert.thumbnail_url.split('/')[-1]
                    storage_service.delete_file(f"alerts/{alert.id}/{object_name}")
                
                # Delete alert record
                db.delete(alert)
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Error deleting alert {alert.id}: {e}")
        
        db.commit()
        db.close()
        
        logger.info(f"Cleaned up {deleted_count} old alerts")
        return f"Cleaned up {deleted_count} old alerts"
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_alerts: {e}")
        raise


@shared_task
def cleanup_storage():
    """Clean up old files in storage"""
    try:
        storage_service = StorageService()
        await storage_service.cleanup_old_files(days_old=30)
        
        logger.info("Storage cleanup completed")
        return "Storage cleanup completed"
        
    except Exception as e:
        logger.error(f"Error in cleanup_storage: {e}")
        raise


@shared_task
def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        import os
        import glob
        
        # Clean up temporary video files
        temp_patterns = [
            "data/clips/*/temp_*",
            "data/clips/*/processing_*",
            "logs/*.log.*"
        ]
        
        deleted_count = 0
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} temporary files")
        return f"Cleaned up {deleted_count} temporary files"
        
    except Exception as e:
        logger.error(f"Error in cleanup_temp_files: {e}")
        raise
