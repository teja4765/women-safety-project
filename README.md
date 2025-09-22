# Safety Detection System

A privacy-aware video analytics system for detecting potentially risky situations in real-time while maintaining strict privacy protections.

## 🎯 Overview

This system is designed to enhance public safety by detecting situations where women may be at risk, such as:
- Lone women at night
- Women surrounded by men
- SOS/distress gestures

The system operates on the principle of "privacy by design" and focuses on situational awareness rather than individual identification.

## ✨ Key Features

### Core Detection Capabilities
- **Person Detection**: Real-time person detection using YOLOv8
- **Gender Classification**: Probabilistic gender estimation with confidence scores
- **Pose Estimation**: SOS gesture detection using MediaPipe
- **Multi-Camera Tracking**: Cross-camera person tracking with ByteTrack
- **Dual Processing Modes**: Live CCTV feeds and video file analysis
- **Batch Processing**: Upload and analyze video files with detailed reporting

### Risk Detection Rules
- **Lone Woman at Night**: Detects isolated women during quiet hours
- **Surrounded by Men**: Identifies women surrounded by multiple men
- **SOS Gestures**: Recognizes distress signals and help requests

### Privacy & Security
- **Face Blurring**: Automatic face blurring in stored clips
- **No Identification**: No facial recognition or personal identification
- **Short Retention**: Automatic deletion of clips after 30 days
- **Encrypted Storage**: All data encrypted at rest and in transit

### Real-time Monitoring
- **Live Dashboard**: Web-based monitoring interface
- **WebSocket Updates**: Real-time alerts and status updates
- **Multi-channel Notifications**: Telegram, Slack, SMS alerts
- **Hotspot Analytics**: Historical risk pattern analysis
- **Processing Mode Management**: Switch between live and batch processing
- **Video File Analysis**: Upload and analyze recorded video files

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RTSP Cameras  │    │  Video Processor │    │  Risk Analyzer  │
│                 │───▶│                 │───▶│                 │
│  - Detection    │    │  - YOLO         │    │  - Lone Woman   │
│  - Tracking     │    │  - ByteTrack    │    │  - Surrounded   │
│  - Gender       │    │  - Gender       │    │  - SOS Gesture  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │  Alert Manager  │    │   Storage       │
│                 │◀───│                 │───▶│                 │
│  - Live View    │    │  - Face Blur    │    │  - MinIO/S3     │
│  - Alerts       │    │  - Clip Gen     │    │  - PostgreSQL   │
│  - Analytics    │    │  - Notifications│    │  - TimescaleDB  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (optional, for better performance)
- Python 3.10+ (for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd safety-detection
   ```

2. **Run setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Download YOLO model**
   ```bash
   mkdir -p models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
   ```

4. **Start the system**
   ```bash
   docker-compose up -d
   ```

5. **Verify installation**
   ```bash
   # Test basic system functionality
   python scripts/test_system.py
   
   # Test video analysis and processing modes
   python scripts/test_video_analysis.py
   ```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:8000 (coming soon)
- **Grafana Monitoring**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

## 📋 Configuration

### Camera Configuration
Edit `config/cameras.yaml` to add your cameras:

```yaml
cameras:
  cam_01:
    id: "cam_01"
    name: "Parking Lot A - North"
    zone_id: "campus_parking_a"
    source: "rtsp://192.168.1.101:554/stream1"
    resolution: [1920, 1080]
    fps: 30
    enabled: true
```

### Zone Configuration
Edit `config/zones.yaml` to define monitoring zones:

```yaml
zones:
  campus_parking_a:
    id: "campus_parking_a"
    name: "Campus Parking Lot A"
    type: "parking"
    risk_level: "high"
    boundaries:
      - [100, 200]
      - [300, 200] 
      - [300, 400]
      - [100, 400]
    quiet_hours:
      start: "22:00"
      end: "06:00"
    thresholds:
      lone_woman_night:
        min_confidence: 0.7
        max_people: 3
        duration_seconds: 10
```

### Environment Variables
Key environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://safety_user:safety_pass@localhost:5432/safety_db

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK_URL=your_webhook_url
```

## 📹 Video Analysis & Processing Modes

### Processing Modes

The system supports three distinct processing modes:

#### 1. Live CCTV Processing (`live_cctv`)
- **Purpose**: Real-time processing of live camera feeds
- **FPS**: 5 FPS for optimal performance
- **Features**: 
  - Real-time alerts and notifications
  - WebSocket updates
  - Temporal analysis with frame buffering
  - Performance monitoring
- **Use Cases**: Active surveillance, real-time monitoring

#### 2. Video File Batch Processing (`video_file_batch`)
- **Purpose**: Thorough analysis of uploaded video files
- **FPS**: 10 FPS for comprehensive analysis
- **Features**:
  - Detailed frame-by-frame analysis
  - Annotated frame generation
  - Comprehensive reporting
  - Batch job management
- **Use Cases**: Incident investigation, forensic analysis, training data generation

#### 3. Video File Real-time Processing (`video_file_realtime`)
- **Purpose**: Simulated live processing of video files
- **FPS**: 5 FPS with real-time playback
- **Features**:
  - Real-time alerts during playback
  - WebSocket updates
  - Temporal analysis
  - Pause/resume capabilities
- **Use Cases**: Testing, demonstration, training scenarios

### Video File Analysis

#### Upload and Process Videos
```bash
# Upload video file for analysis
curl -X POST "http://localhost:8000/api/v1/video-analysis/upload" \
  -F "file=@incident_video.mp4" \
  -F "zone_id=campus_parking_a" \
  -F "processing_mode=batch"

# Process existing video file
curl -X POST "http://localhost:8000/api/v1/video-analysis/process-file" \
  -F "video_path=/path/to/video.mp4" \
  -F "zone_id=campus_parking_a" \
  -F "processing_mode=batch"
```

#### Monitor Analysis Progress
```bash
# Get all analysis jobs
curl "http://localhost:8000/api/v1/video-analysis/jobs"

# Get specific job details
curl "http://localhost:8000/api/v1/video-analysis/jobs/{job_id}"

# Get analysis results
curl "http://localhost:8000/api/v1/video-analysis/jobs/{job_id}/results"
```

#### Download Results
```bash
# Download complete analysis results as ZIP
curl "http://localhost:8000/api/v1/video-analysis/jobs/{job_id}/download" \
  -o analysis_results.zip
```

### Processing Mode Management

#### Auto-Select Processing Mode
```bash
# Auto-select mode for RTSP stream
curl "http://localhost:8000/api/v1/processing-modes/auto-select?input_type=rtsp_url"

# Auto-select mode for file upload
curl "http://localhost:8000/api/v1/processing-modes/auto-select?input_type=file_upload"
```

#### Switch Camera Processing Mode
```bash
# Switch camera to batch processing mode
curl -X POST "http://localhost:8000/api/v1/processing-modes/camera/cam_01/switch" \
  -H "Content-Type: application/json" \
  -d '{"mode": "video_file_batch", "reason": "Incident investigation"}'
```

#### Get Processing Mode Statistics
```bash
# Get mode usage statistics
curl "http://localhost:8000/api/v1/processing-modes/statistics"
```

### Supported Video Formats

#### Live CCTV
- RTSP streams (`rtsp://`)
- HTTP streams (`http://`)
- UDP streams (`udp://`)
- Webcam devices (device index)

#### Video Files
- MP4 (`.mp4`)
- AVI (`.avi`)
- MOV (`.mov`)
- MKV (`.mkv`)
- FLV (`.flv`)

### Analysis Output

#### Batch Processing Results
- **Annotated Frames**: Frames with detection overlays
- **Alert Reports**: JSON files with detected risks
- **Processing Report**: Summary of analysis results
- **Video Clips**: Short clips of detected incidents (with face blurring)

#### Real-time Processing
- **Live Alerts**: Immediate notifications via WebSocket
- **Status Updates**: Real-time processing metrics
- **Performance Data**: FPS, latency, and resource usage

## 🔧 Development

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start services**
   ```bash
   docker-compose up -d postgres redis minio
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Run tests**
   ```bash
   python scripts/test_system.py
   pytest tests/
   ```

### Project Structure
```
safety-detection/
├── app/                    # Main application
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   ├── models/            # Database models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic
│   └── websocket/         # WebSocket handling
├── config/                # Configuration files
├── data/                  # Data storage
├── models/                # ML models
├── scripts/               # Utility scripts
├── sql/                   # Database scripts
└── tests/                 # Test files
```

## 📊 API Documentation

### Core Endpoints

#### Health Check
```bash
GET /api/v1/health
GET /api/v1/health/detailed
GET /api/v1/health/cameras
```

#### Alerts
```bash
GET /api/v1/alerts                    # List alerts
GET /api/v1/alerts/{id}              # Get specific alert
POST /api/v1/alerts/{id}/acknowledge # Acknowledge alert
POST /api/v1/alerts/{id}/escalate    # Escalate alert
GET /api/v1/alerts/stats/summary     # Alert statistics
```

#### Cameras
```bash
GET /api/v1/cameras                  # List cameras
GET /api/v1/cameras/{id}            # Get camera details
GET /api/v1/cameras/{id}/status     # Camera status
POST /api/v1/cameras/{id}/enable    # Enable camera
POST /api/v1/cameras/{id}/disable   # Disable camera
```

#### Analytics
```bash
GET /api/v1/analytics/hotspots           # Hotspot data
GET /api/v1/analytics/gender-distribution # Gender analytics
GET /api/v1/analytics/system-metrics     # System metrics
GET /api/v1/analytics/summary           # Analytics summary
```

#### Video Analysis
```bash
POST /api/v1/video-analysis/upload       # Upload video for analysis
POST /api/v1/video-analysis/process-file # Process existing video file
GET /api/v1/video-analysis/jobs          # List analysis jobs
GET /api/v1/video-analysis/jobs/{id}     # Get job details
GET /api/v1/video-analysis/jobs/{id}/results # Get analysis results
GET /api/v1/video-analysis/status        # System status
```

#### Processing Modes
```bash
GET /api/v1/processing-modes             # Available processing modes
GET /api/v1/processing-modes/active      # Active modes for cameras
GET /api/v1/processing-modes/auto-select # Auto-select mode for input
POST /api/v1/processing-modes/camera/{id}/switch # Switch camera mode
GET /api/v1/processing-modes/statistics  # Mode usage statistics
```

### WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🔒 Privacy & Ethics

### Privacy Principles
- **No Personal Identification**: No facial recognition or identity tracking
- **Data Minimization**: Only event clips stored, not continuous video
- **Short Retention**: Automatic deletion after 30 days
- **Face Blurring**: All stored clips have faces automatically blurred

### Bias Control
- **Probabilistic Classification**: Gender estimates include confidence scores
- **Subgroup Monitoring**: Performance tracked across demographic groups
- **Human Oversight**: All alerts require human review
- **Transparency**: Confidence scores provided for all decisions

### Compliance
- **Purpose Limitation**: Data used only for stated safety purposes
- **Lawful Basis**: Public safety and legitimate interest
- **Data Subject Rights**: Access, rectification, and erasure rights
- **Audit Trail**: Complete logging of all data access

## 📈 Monitoring & Analytics

### Real-time Metrics
- Camera status and performance
- Detection rates and accuracy
- Alert frequency and types
- System resource usage

### Historical Analytics
- Hotspot identification
- Risk pattern analysis
- Performance trends
- False positive rates

### Dashboards
- **Grafana**: System monitoring and metrics
- **Web Dashboard**: Real-time alerts and camera views
- **API**: Programmatic access to all data

## 🚨 Alert Types

### Lone Woman at Night
- **Trigger**: Woman alone during quiet hours
- **Confidence**: Based on gender classification
- **Duration**: Must persist for configurable time
- **Severity**: Higher at night, weekends, holidays

### Surrounded by Men
- **Trigger**: Woman surrounded by multiple men
- **Distance**: Configurable proximity threshold
- **Duration**: Must persist for configurable time
- **Severity**: Based on number of men and proximity

### SOS Gesture
- **Trigger**: Hand-raising and waving gestures
- **Detection**: Pose estimation and temporal analysis
- **Duration**: Must persist for configurable time
- **Context**: Suppressed near entrances or festive areas

## 🔧 Troubleshooting

### Common Issues

#### Camera Connection Failed
```bash
# Check camera URL
curl -I rtsp://your-camera-ip:554/stream1

# Test with VLC
vlc rtsp://your-camera-ip:554/stream1
```

#### Model Loading Error
```bash
# Download missing models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

#### Database Connection Error
```bash
# Check database status
docker-compose ps postgres

# View logs
docker-compose logs postgres
```

#### High CPU Usage
- Reduce processing FPS in configuration
- Use GPU acceleration if available
- Optimize detection confidence thresholds

### Performance Tuning

#### For Edge Devices (Jetson)
```yaml
# config/zones.yaml
system:
  fps: 3  # Reduce FPS
  detection_confidence_threshold: 0.6  # Higher threshold
```

#### For High-Performance Servers
```yaml
# config/zones.yaml
system:
  fps: 10  # Higher FPS
  detection_confidence_threshold: 0.4  # Lower threshold
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation
- Respect privacy and ethics principles

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for person detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking
- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [TimescaleDB](https://www.timescale.com/) for time-series data

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

---

**⚠️ Important**: This system is designed for legitimate safety purposes only. Users must comply with all applicable laws and regulations regarding privacy, surveillance, and data protection in their jurisdiction.
