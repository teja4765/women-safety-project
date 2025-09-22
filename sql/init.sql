-- Database initialization script for Safety Detection System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Create database if it doesn't exist (this would be done at the database level)
-- CREATE DATABASE safety_db;

-- Create tables (these will be created by SQLAlchemy, but here for reference)

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR PRIMARY KEY,
    type VARCHAR NOT NULL,
    camera_id VARCHAR NOT NULL,
    zone_id VARCHAR NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    confidence FLOAT NOT NULL,
    severity FLOAT NOT NULL,
    description TEXT,
    metadata JSONB,
    clip_url VARCHAR,
    thumbnail_url VARCHAR,
    status VARCHAR DEFAULT 'pending',
    acknowledged_by VARCHAR,
    acknowledged_at TIMESTAMP,
    escalated_by VARCHAR,
    escalated_at TIMESTAMP,
    operator_feedback TEXT,
    false_positive BOOLEAN DEFAULT FALSE
);

-- Create indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type);
CREATE INDEX IF NOT EXISTS idx_alerts_camera_id ON alerts(camera_id);
CREATE INDEX IF NOT EXISTS idx_alerts_zone_id ON alerts(zone_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_start_time ON alerts(start_time);

-- Convert alerts to hypertable for time-series data
SELECT create_hypertable('alerts', 'created_at', if_not_exists => TRUE);

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    zone_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    resolution_width INTEGER DEFAULT 1920,
    resolution_height INTEGER DEFAULT 1080,
    fps INTEGER DEFAULT 30,
    position_x FLOAT,
    position_y FLOAT,
    calibration_data JSONB,
    enabled BOOLEAN DEFAULT TRUE,
    status VARCHAR DEFAULT 'offline',
    last_heartbeat TIMESTAMP,
    processing_fps FLOAT DEFAULT 0.0,
    detection_count INTEGER DEFAULT 0,
    last_detection TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for cameras
CREATE INDEX IF NOT EXISTS idx_cameras_zone_id ON cameras(zone_id);
CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras(enabled);
CREATE INDEX IF NOT EXISTS idx_cameras_status ON cameras(status);

-- Zones table
CREATE TABLE IF NOT EXISTS zones (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    type VARCHAR NOT NULL,
    risk_level VARCHAR NOT NULL,
    boundaries JSONB NOT NULL,
    thresholds JSONB NOT NULL,
    quiet_hours JSONB,
    camera_ids JSONB,
    total_alerts INTEGER DEFAULT 0,
    last_alert TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Hotspots table
CREATE TABLE IF NOT EXISTS hotspots (
    id VARCHAR PRIMARY KEY,
    zone_id VARCHAR NOT NULL,
    geohash VARCHAR NOT NULL,
    hour_of_week INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    total_alerts INTEGER DEFAULT 0,
    lone_woman_alerts INTEGER DEFAULT 0,
    surrounded_alerts INTEGER DEFAULT 0,
    sos_alerts INTEGER DEFAULT 0,
    risk_score FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for hotspots
CREATE INDEX IF NOT EXISTS idx_hotspots_zone_id ON hotspots(zone_id);
CREATE INDEX IF NOT EXISTS idx_hotspots_geohash ON hotspots(geohash);
CREATE INDEX IF NOT EXISTS idx_hotspots_hour_of_week ON hotspots(hour_of_week);
CREATE INDEX IF NOT EXISTS idx_hotspots_date ON hotspots(date);
CREATE INDEX IF NOT EXISTS idx_hotspots_zone_hour_date ON hotspots(zone_id, hour_of_week, date);

-- Convert hotspots to hypertable
SELECT create_hypertable('hotspots', 'date', if_not_exists => TRUE);

-- Gender distributions table
CREATE TABLE IF NOT EXISTS gender_distributions (
    id VARCHAR PRIMARY KEY,
    camera_id VARCHAR NOT NULL,
    zone_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    expected_females FLOAT DEFAULT 0.0,
    expected_males FLOAT DEFAULT 0.0,
    total_people INTEGER DEFAULT 0,
    avg_female_confidence FLOAT DEFAULT 0.0,
    avg_male_confidence FLOAT DEFAULT 0.0,
    is_night_time BOOLEAN DEFAULT FALSE,
    light_level FLOAT
);

-- Create indexes for gender distributions
CREATE INDEX IF NOT EXISTS idx_gender_dist_camera_id ON gender_distributions(camera_id);
CREATE INDEX IF NOT EXISTS idx_gender_dist_zone_id ON gender_distributions(zone_id);
CREATE INDEX IF NOT EXISTS idx_gender_dist_timestamp ON gender_distributions(timestamp);

-- Convert gender distributions to hypertable
SELECT create_hypertable('gender_distributions', 'timestamp', if_not_exists => TRUE);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_cameras INTEGER DEFAULT 0,
    active_cameras INTEGER DEFAULT 0,
    processing_fps FLOAT DEFAULT 0.0,
    detection_latency_ms FLOAT DEFAULT 0.0,
    alerts_per_hour FLOAT DEFAULT 0.0,
    false_positive_rate FLOAT DEFAULT 0.0,
    operator_response_time_minutes FLOAT DEFAULT 0.0,
    cpu_usage FLOAT DEFAULT 0.0,
    memory_usage FLOAT DEFAULT 0.0,
    gpu_usage FLOAT DEFAULT 0.0
);

-- Create indexes for system metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Convert system metrics to hypertable
SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);

-- Create retention policies
-- Keep alerts for 30 days
SELECT add_retention_policy('alerts', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep gender distributions for 7 days
SELECT add_retention_policy('gender_distributions', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep system metrics for 30 days
SELECT add_retention_policy('system_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep hotspots for 1 year (for long-term analytics)
SELECT add_retention_policy('hotspots', INTERVAL '1 year', if_not_exists => TRUE);

-- Insert initial data
INSERT INTO zones (id, name, type, risk_level, boundaries, thresholds, quiet_hours, camera_ids) VALUES
('campus_parking_a', 'Campus Parking Lot A', 'parking', 'high', 
 '[[100, 200], [300, 200], [300, 400], [100, 400]]',
 '{"lone_woman_night": {"min_confidence": 0.7, "max_people": 3, "duration_seconds": 10}, "surrounded": {"min_female_confidence": 0.6, "min_male_confidence": 0.6, "min_males": 3, "max_distance_meters": 2.0, "duration_seconds": 15}, "sos_gesture": {"min_confidence": 0.8, "duration_seconds": 5}}',
 '{"start": "22:00", "end": "06:00"}',
 '["cam_01", "cam_02"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO zones (id, name, type, risk_level, boundaries, thresholds, quiet_hours, camera_ids) VALUES
('campus_quad', 'Main Campus Quad', 'open_space', 'medium',
 '[[50, 50], [450, 50], [450, 350], [50, 350]]',
 '{"lone_woman_night": {"min_confidence": 0.7, "max_people": 2, "duration_seconds": 8}, "surrounded": {"min_female_confidence": 0.6, "min_male_confidence": 0.6, "min_males": 4, "max_distance_meters": 1.5, "duration_seconds": 12}, "sos_gesture": {"min_confidence": 0.8, "duration_seconds": 5}}',
 '{"start": "23:00", "end": "05:00"}',
 '["cam_03", "cam_04"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO zones (id, name, type, risk_level, boundaries, thresholds, quiet_hours, camera_ids) VALUES
('downtown_park', 'Downtown Central Park', 'park', 'high',
 '[[0, 0], [500, 0], [500, 300], [0, 300]]',
 '{"lone_woman_night": {"min_confidence": 0.7, "max_people": 2, "duration_seconds": 10}, "surrounded": {"min_female_confidence": 0.6, "min_male_confidence": 0.6, "min_males": 3, "max_distance_meters": 2.0, "duration_seconds": 15}, "sos_gesture": {"min_confidence": 0.8, "duration_seconds": 5}}',
 '{"start": "21:00", "end": "07:00"}',
 '["cam_05", "cam_06", "cam_07"]')
ON CONFLICT (id) DO NOTHING;

-- Insert initial cameras
INSERT INTO cameras (id, name, zone_id, source, resolution_width, resolution_height, fps, position_x, position_y, enabled) VALUES
('cam_01', 'Parking Lot A - North', 'campus_parking_a', 'rtsp://192.168.1.101:554/stream1', 1920, 1080, 30, 200, 200, true),
('cam_02', 'Parking Lot A - South', 'campus_parking_a', 'rtsp://192.168.1.102:554/stream1', 1920, 1080, 30, 200, 350, true),
('cam_03', 'Quad - East', 'campus_quad', 'rtsp://192.168.1.103:554/stream1', 1920, 1080, 30, 250, 200, true),
('cam_04', 'Quad - West', 'campus_quad', 'rtsp://192.168.1.104:554/stream1', 1920, 1080, 30, 250, 200, true),
('cam_05', 'Park - Entrance', 'downtown_park', 'rtsp://192.168.1.105:554/stream1', 1920, 1080, 30, 250, 50, true),
('cam_06', 'Park - Center', 'downtown_park', 'rtsp://192.168.1.106:554/stream1', 1920, 1080, 30, 250, 150, true),
('cam_07', 'Park - Playground', 'downtown_park', 'rtsp://192.168.1.107:554/stream1', 1920, 1080, 30, 400, 200, true)
ON CONFLICT (id) DO NOTHING;

-- Create a test camera for development
INSERT INTO cameras (id, name, zone_id, source, resolution_width, resolution_height, fps, position_x, position_y, enabled) VALUES
('test_cam', 'Test Camera', 'campus_parking_a', '0', 640, 480, 30, 200, 200, true)
ON CONFLICT (id) DO NOTHING;

COMMIT;
