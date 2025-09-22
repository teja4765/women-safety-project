"""
WebSocket connection manager for real-time updates
"""

import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket
import asyncio

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_subscriptions: Dict[WebSocket, List[str]] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_subscriptions[websocket] = []
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_subscriptions:
            del self.connection_subscriptions[websocket]
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        # Create tasks for all connections
        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self._send_to_connection(connection, message))
        
        # Send to all connections concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_connection(self, websocket: WebSocket, message: str):
        """Send message to a single connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending to connection: {e}")
            self.disconnect(websocket)
    
    async def send_alert_update(self, alert: Dict[str, Any]):
        """Send alert update to all connected clients"""
        message = {
            "type": "alert",
            "data": alert
        }
        await self.broadcast(json.dumps(message, default=str))
    
    async def send_camera_status_update(self, camera_id: str, status: Dict[str, Any]):
        """Send camera status update to all connected clients"""
        message = {
            "type": "camera_status",
            "camera_id": camera_id,
            "data": status
        }
        await self.broadcast(json.dumps(message, default=str))
    
    async def send_gender_distribution_update(self, camera_id: str, distribution: Dict[str, Any]):
        """Send gender distribution update to all connected clients"""
        message = {
            "type": "gender_distribution",
            "camera_id": camera_id,
            "data": distribution
        }
        await self.broadcast(json.dumps(message, default=str))
    
    async def send_system_metrics_update(self, metrics: Dict[str, Any]):
        """Send system metrics update to all connected clients"""
        message = {
            "type": "system_metrics",
            "data": metrics
        }
        await self.broadcast(json.dumps(message, default=str))
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "active_connections": len(self.active_connections),
            "subscriptions": {
                websocket: subscriptions 
                for websocket, subscriptions in self.connection_subscriptions.items()
            }
        }
