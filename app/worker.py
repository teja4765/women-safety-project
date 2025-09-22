"""
Celery worker for background tasks
"""

from celery import Celery
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "safety_detection",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Import tasks
from app.tasks import cleanup_tasks, analytics_tasks, notification_tasks

if __name__ == "__main__":
    celery_app.start()
