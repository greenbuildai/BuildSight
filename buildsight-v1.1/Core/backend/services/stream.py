import asyncio
import json
import logging
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from ..schemas import AlertEvent

logger = logging.getLogger(__name__)

class StreamService:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_alert(self, alert: AlertEvent):
        if not self.active_connections:
            return
        
        message = alert.model_dump_json()
        disconnected_sockets = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                disconnected_sockets.append(connection)
        
        for ds in disconnected_sockets:
            self.disconnect(ds)

# Global instance
stream_service = StreamService()
