"""
Analytics background tasks
"""

from celery import shared_task
from datetime import datetime, timedelta
import logging

from app.core.database import SessionLocal
from app.models.analytics import Hotspot, GenderDistribution, SystemMetrics
from app.models.alert import Alert
from app.models.camera import Camera

logger = logging.getLogger(__name__)


@shared_task
def generate_hotspot_analytics():
    """Generate hotspot analytics from recent alerts"""
    try:
        db = SessionLocal()
        
        # Get alerts from the last 24 hours
        start_time = datetime.now() - timedelta(hours=24)
        alerts = db.query(Alert).filter(
            Alert.created_at >= start_time
        ).all()
        
        # Group alerts by zone and hour
        hotspot_data = {}
        for alert in alerts:
            zone_id = alert.zone_id
            hour_of_week = alert.created_at.weekday() * 24 + alert.created_at.hour
            
            key = (zone_id, hour_of_week)
            if key not in hotspot_data:
                hotspot_data[key] = {
                    'zone_id': zone_id,
                    'hour_of_week': hour_of_week,
                    'date': alert.created_at.date(),
                    'total_alerts': 0,
                    'lone_woman_alerts': 0,
                    'surrounded_alerts': 0,
                    'sos_alerts': 0
                }
            
            hotspot_data[key]['total_alerts'] += 1
            
            if alert.type == 'LONE_WOMAN_NIGHT':
                hotspot_data[key]['lone_woman_alerts'] += 1
            elif alert.type == 'SURROUNDED':
                hotspot_data[key]['surrounded_alerts'] += 1
            elif alert.type == 'SOS_GESTURE':
                hotspot_data[key]['sos_alerts'] += 1
        
        # Create or update hotspot records
        created_count = 0
        for (zone_id, hour_of_week), data in hotspot_data.items():
            # Calculate risk score
            risk_score = min(data['total_alerts'] / 10.0, 1.0)  # Normalize to 0-1
            
            # Check if hotspot already exists
            existing = db.query(Hotspot).filter(
                Hotspot.zone_id == zone_id,
                Hotspot.hour_of_week == hour_of_week,
                Hotspot.date == data['date']
            ).first()
            
            if existing:
                # Update existing
                existing.total_alerts = data['total_alerts']
                existing.lone_woman_alerts = data['lone_woman_alerts']
                existing.surrounded_alerts = data['surrounded_alerts']
                existing.sos_alerts = data['sos_alerts']
                existing.risk_score = risk_score
                existing.updated_at = datetime.now()
            else:
                # Create new
                hotspot = Hotspot(
                    id=f"{zone_id}_{hour_of_week}_{data['date']}",
                    zone_id=zone_id,
                    geohash="",  # Would be calculated from zone boundaries
                    hour_of_week=hour_of_week,
                    date=data['date'],
                    total_alerts=data['total_alerts'],
                    lone_woman_alerts=data['lone_woman_alerts'],
                    surrounded_alerts=data['surrounded_alerts'],
                    sos_alerts=data['sos_alerts'],
                    risk_score=risk_score,
                    confidence=0.8
                )
                db.add(hotspot)
                created_count += 1
        
        db.commit()
        db.close()
        
        logger.info(f"Generated hotspot analytics: {created_count} new hotspots")
        return f"Generated {created_count} hotspot records"
        
    except Exception as e:
        logger.error(f"Error in generate_hotspot_analytics: {e}")
        raise


@shared_task
def generate_system_metrics():
    """Generate system performance metrics"""
    try:
        db = SessionLocal()
        
        # Get current system state
        total_cameras = db.query(Camera).count()
        active_cameras = db.query(Camera).filter(Camera.status == "online").count()
        
        # Get recent alerts for rate calculation
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = db.query(Alert).filter(
            Alert.created_at >= hour_ago
        ).count()
        
        # Get false positive rate
        total_resolved = db.query(Alert).filter(
            Alert.status == "resolved"
        ).count()
        false_positives = db.query(Alert).filter(
            Alert.status == "resolved",
            Alert.false_positive == True
        ).count()
        
        false_positive_rate = false_positives / max(total_resolved, 1)
        
        # Create system metrics record
        metrics = SystemMetrics(
            id=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            total_cameras=total_cameras,
            active_cameras=active_cameras,
            processing_fps=5.0,  # Would be calculated from actual processing
            detection_latency_ms=100.0,  # Would be measured
            alerts_per_hour=recent_alerts,
            false_positive_rate=false_positive_rate,
            operator_response_time_minutes=15.0,  # Would be calculated
            cpu_usage=0.0,  # Would be measured
            memory_usage=0.0,  # Would be measured
            gpu_usage=0.0  # Would be measured
        )
        
        db.add(metrics)
        db.commit()
        db.close()
        
        logger.info("Generated system metrics")
        return "System metrics generated"
        
    except Exception as e:
        logger.error(f"Error in generate_system_metrics: {e}")
        raise


@shared_task
def generate_weekly_report():
    """Generate weekly analytics report"""
    try:
        db = SessionLocal()
        
        # Get data for the past week
        week_ago = datetime.now() - timedelta(days=7)
        
        # Get alert statistics
        total_alerts = db.query(Alert).filter(
            Alert.created_at >= week_ago
        ).count()
        
        alerts_by_type = {}
        for alert_type in ['LONE_WOMAN_NIGHT', 'SURROUNDED', 'SOS_GESTURE']:
            count = db.query(Alert).filter(
                Alert.created_at >= week_ago,
                Alert.type == alert_type
            ).count()
            alerts_by_type[alert_type] = count
        
        # Get top hotspots
        top_hotspots = db.query(Hotspot).filter(
            Hotspot.date >= week_ago
        ).order_by(Hotspot.risk_score.desc()).limit(10).all()
        
        # Generate report (would be saved to file or sent via email)
        report = {
            'period': f"{week_ago.date()} to {datetime.now().date()}",
            'total_alerts': total_alerts,
            'alerts_by_type': alerts_by_type,
            'top_hotspots': [
                {
                    'zone_id': h.zone_id,
                    'risk_score': h.risk_score,
                    'total_alerts': h.total_alerts
                }
                for h in top_hotspots
            ]
        }
        
        logger.info(f"Weekly report generated: {total_alerts} alerts")
        return f"Weekly report: {total_alerts} alerts"
        
    except Exception as e:
        logger.error(f"Error in generate_weekly_report: {e}")
        raise
