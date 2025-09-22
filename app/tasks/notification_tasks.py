"""
Notification background tasks
"""

from celery import shared_task
import logging

from app.services.notification import NotificationService

logger = logging.getLogger(__name__)


@shared_task
def send_alert_notification(alert_data):
    """Send alert notification asynchronously"""
    try:
        notification_service = NotificationService()
        await notification_service.send_alert_notification(alert_data)
        
        logger.info(f"Alert notification sent for {alert_data.get('id')}")
        return f"Notification sent for alert {alert_data.get('id')}"
        
    except Exception as e:
        logger.error(f"Error sending alert notification: {e}")
        raise


@shared_task
def send_system_notification(message, level="info"):
    """Send system notification asynchronously"""
    try:
        notification_service = NotificationService()
        await notification_service.send_system_notification(message, level)
        
        logger.info(f"System notification sent: {message}")
        return f"System notification sent: {message}"
        
    except Exception as e:
        logger.error(f"Error sending system notification: {e}")
        raise


@shared_task
def send_daily_summary():
    """Send daily summary notification"""
    try:
        from datetime import datetime, timedelta
        from app.core.database import SessionLocal
        from app.models.alert import Alert
        
        db = SessionLocal()
        
        # Get yesterday's alerts
        yesterday = datetime.now() - timedelta(days=1)
        start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        alerts = db.query(Alert).filter(
            Alert.created_at >= start_of_day,
            Alert.created_at <= end_of_day
        ).all()
        
        # Generate summary
        total_alerts = len(alerts)
        alerts_by_type = {}
        for alert in alerts:
            alert_type = alert.type
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Format message
        message = f"Daily Summary - {yesterday.strftime('%Y-%m-%d')}\n"
        message += f"Total Alerts: {total_alerts}\n"
        message += "By Type:\n"
        for alert_type, count in alerts_by_type.items():
            message += f"  {alert_type}: {count}\n"
        
        # Send notification
        notification_service = NotificationService()
        await notification_service.send_system_notification(message, "info")
        
        db.close()
        
        logger.info(f"Daily summary sent: {total_alerts} alerts")
        return f"Daily summary sent: {total_alerts} alerts"
        
    except Exception as e:
        logger.error(f"Error sending daily summary: {e}")
        raise


@shared_task
def send_weekly_report():
    """Send weekly analytics report"""
    try:
        from datetime import datetime, timedelta
        from app.core.database import SessionLocal
        from app.models.alert import Alert
        from app.models.analytics import Hotspot
        
        db = SessionLocal()
        
        # Get last week's data
        week_ago = datetime.now() - timedelta(days=7)
        
        # Get alert statistics
        total_alerts = db.query(Alert).filter(
            Alert.created_at >= week_ago
        ).count()
        
        # Get top hotspots
        top_hotspots = db.query(Hotspot).filter(
            Hotspot.date >= week_ago
        ).order_by(Hotspot.risk_score.desc()).limit(5).all()
        
        # Format message
        message = f"Weekly Report - {week_ago.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}\n"
        message += f"Total Alerts: {total_alerts}\n"
        message += "Top Risk Areas:\n"
        for hotspot in top_hotspots:
            message += f"  {hotspot.zone_id}: {hotspot.risk_score:.2f} risk score\n"
        
        # Send notification
        notification_service = NotificationService()
        await notification_service.send_system_notification(message, "info")
        
        db.close()
        
        logger.info(f"Weekly report sent: {total_alerts} alerts")
        return f"Weekly report sent: {total_alerts} alerts"
        
    except Exception as e:
        logger.error(f"Error sending weekly report: {e}")
        raise
