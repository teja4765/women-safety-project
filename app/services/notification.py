"""
Notification service for sending alerts via various channels
"""

import asyncio
import requests
import logging
from typing import Dict, Optional
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class NotificationService:
    """Notification service for sending alerts"""
    
    def __init__(self):
        self.telegram_bot_token = settings.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = settings.TELEGRAM_CHAT_ID
        self.slack_webhook_url = settings.SLACK_WEBHOOK_URL
        
    async def initialize(self):
        """Initialize notification service"""
        logger.info("Notification service initialized")
    
    async def send_alert_notification(self, alert: Dict):
        """
        Send alert notification via configured channels
        
        Args:
            alert: Alert dictionary
        """
        try:
            # Send to Telegram if configured
            if self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram_alert(alert)
            
            # Send to Slack if configured
            if self.slack_webhook_url:
                await self._send_slack_alert(alert)
            
            # Log notification
            logger.info(f"Alert notification sent for {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def _send_telegram_alert(self, alert: Dict):
        """Send alert to Telegram"""
        try:
            # Format message
            message = self._format_telegram_message(alert)
            
            # Send message
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            # Add thumbnail if available
            if alert.get('thumbnail_url'):
                # Send photo with caption
                photo_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
                photo_data = {
                    'chat_id': self.telegram_chat_id,
                    'photo': alert['thumbnail_url'],
                    'caption': message,
                    'parse_mode': 'HTML'
                }
                
                response = requests.post(photo_url, data=photo_data, timeout=10)
            else:
                response = requests.post(url, data=data, timeout=10)
            
            response.raise_for_status()
            logger.info("Telegram alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def _send_slack_alert(self, alert: Dict):
        """Send alert to Slack"""
        try:
            # Format message
            message = self._format_slack_message(alert)
            
            # Send webhook
            response = requests.post(
                self.slack_webhook_url,
                json=message,
                timeout=10
            )
            
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _format_telegram_message(self, alert: Dict) -> str:
        """Format alert message for Telegram"""
        severity_emoji = {
            'LONE_WOMAN_NIGHT': '🌙',
            'SURROUNDED': '⚠️',
            'SOS_GESTURE': '🆘'
        }
        
        emoji = severity_emoji.get(alert['type'], '🚨')
        
        message = f"""
{emoji} <b>Safety Alert</b>

<b>Type:</b> {alert['type'].replace('_', ' ').title()}
<b>Location:</b> {alert['zone_id']} (Camera: {alert['camera_id']})
<b>Confidence:</b> {alert['confidence']:.2f}
<b>Severity:</b> {alert['severity']:.2f}
<b>Time:</b> {alert['start_time'].strftime('%Y-%m-%d %H:%M:%S')}

<b>Description:</b> {alert['description']}

<b>Alert ID:</b> {alert['id']}
        """.strip()
        
        return message
    
    def _format_slack_message(self, alert: Dict) -> Dict:
        """Format alert message for Slack"""
        severity_color = {
            'LONE_WOMAN_NIGHT': '#FFA500',  # Orange
            'SURROUNDED': '#FF0000',        # Red
            'SOS_GESTURE': '#8B0000'        # Dark Red
        }
        
        color = severity_color.get(alert['type'], '#FF0000')
        
        # Create Slack message
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 Safety Alert: {alert['type'].replace('_', ' ').title()}",
                    "fields": [
                        {
                            "title": "Location",
                            "value": f"{alert['zone_id']} (Camera: {alert['camera_id']})",
                            "short": True
                        },
                        {
                            "title": "Confidence",
                            "value": f"{alert['confidence']:.2f}",
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": f"{alert['severity']:.2f}",
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        },
                        {
                            "title": "Description",
                            "value": alert['description'],
                            "short": False
                        }
                    ],
                    "footer": f"Alert ID: {alert['id']}",
                    "ts": int(alert['start_time'].timestamp())
                }
            ]
        }
        
        # Add thumbnail if available
        if alert.get('thumbnail_url'):
            message["attachments"][0]["image_url"] = alert['thumbnail_url']
        
        return message
    
    async def send_system_notification(self, message: str, level: str = "info"):
        """
        Send system notification
        
        Args:
            message: Notification message
            level: Notification level (info, warning, error)
        """
        try:
            # Format system message
            system_alert = {
                'id': f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'type': 'SYSTEM',
                'camera_id': 'system',
                'zone_id': 'system',
                'start_time': datetime.now(),
                'confidence': 1.0,
                'severity': 0.5 if level == 'warning' else 0.8 if level == 'error' else 0.2,
                'description': f"System {level}: {message}",
                'metadata': {'level': level}
            }
            
            # Send notifications
            if self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram_alert(system_alert)
            
            if self.slack_webhook_url:
                await self._send_slack_alert(system_alert)
            
            logger.info(f"System notification sent: {message}")
            
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")
    
    def get_notification_config(self) -> Dict:
        """Get notification service configuration"""
        return {
            'telegram_configured': bool(self.telegram_bot_token and self.telegram_chat_id),
            'slack_configured': bool(self.slack_webhook_url),
            'channels': {
                'telegram': bool(self.telegram_bot_token and self.telegram_chat_id),
                'slack': bool(self.slack_webhook_url)
            }
        }
