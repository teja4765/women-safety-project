"""
Configuration management for Safety Detection System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Safety Detection System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./safety_detection.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # MinIO/S3 Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "safety-clips"
    MINIO_SECURE: bool = False
    
    # Video Processing
    DEFAULT_FPS: int = 5
    CLIP_DURATION_SECONDS: int = 8
    FACE_BLUR_RADIUS: int = 15
    
    # Alert Settings
    ALERT_COOLDOWN_SECONDS: int = 300
    MAX_ALERTS_PER_HOUR: int = 10
    
    # Model Paths
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"
    GENDER_MODEL_PATH: str = "models/gender_classifier.pth"
    POSE_MODEL_PATH: str = "models/pose_estimator.pth"
    
    # Notification
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
