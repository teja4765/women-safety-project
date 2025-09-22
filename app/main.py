"""
Main FastAPI application for Safety Detection System
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.api import alerts, cameras, zones, analytics, health, video_analysis, processing_modes
from app.api import image_analysis
from app.core.config import settings
from app.core.database import init_db
from app.services.alert_manager import AlertManager
from app.websocket.connection_manager import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services (live processing disabled for video-analysis-only demo)
alert_manager = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global alert_manager
    
    logger.info("Starting Safety Detection System...")
    
    # Initialize database
    await init_db()
    
    # Initialize services needed for video analysis API only
    alert_manager = AlertManager()
    
    logger.info("System started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Safety Detection System...")
    # No live processing to stop in video-analysis-only mode
    logger.info("System shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Safety Detection System",
    description="Privacy-aware video analytics for detecting potentially risky situations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(cameras.router, prefix="/api/v1", tags=["cameras"])
app.include_router(zones.router, prefix="/api/v1", tags=["zones"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(video_analysis.router, prefix="/api/v1", tags=["video-analysis"])
app.include_router(processing_modes.router, prefix="/api/v1", tags=["processing-modes"])
app.include_router(image_analysis.router, prefix="/api/v1", tags=["image-analysis"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for now - can be extended for client commands
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Safety Detection System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
