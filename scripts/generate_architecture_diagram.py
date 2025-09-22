#!/usr/bin/env python3
"""
Generate architecture diagram for Safety Detection System
"""

def generate_architecture_diagram():
    """Generate Mermaid diagram code for the system architecture"""
    
    diagram = """
graph TB
    %% Input Sources
    subgraph "Input Sources"
        RTSP[RTSP Cameras]
        UPLOAD[Video Uploads]
        LIVE[Live Streams]
    end
    
    %% Core Processing Pipeline
    subgraph "AI Processing Pipeline"
        DETECT[Person Detection<br/>YOLOv8]
        TRACK[Multi-Object Tracking<br/>ByteTrack]
        GENDER[Gender Classification<br/>CNN]
        POSE[Pose Estimation<br/>MediaPipe]
        VIOLENCE[Violence Detection<br/>I3D]
    end
    
    %% Risk Analysis
    subgraph "Risk Analysis Engine"
        RULES[Safety Rules Engine]
        CONTEXT[Contextual Analysis]
        TEMPORAL[Temporal Buffering]
    end
    
    %% Processing Modes
    subgraph "Processing Modes"
        LIVE_PROC[Live CCTV<br/>5 FPS]
        BATCH_PROC[Video File Batch<br/>10 FPS]
        REALTIME_PROC[Video File Real-time<br/>5 FPS]
    end
    
    %% Privacy & Security
    subgraph "Privacy Protection"
        FACE_BLUR[Face Blurring]
        RETENTION[30-day Retention]
        ENCRYPT[Encrypted Storage]
    end
    
    %% Storage Layer
    subgraph "Storage & Data"
        MINIO[MinIO Object Storage]
        POSTGRES[(PostgreSQL<br/>Metadata)]
        TIMESCALE[(TimescaleDB<br/>Time Series)]
        REDIS[(Redis Cache)]
    end
    
    %% Alert System
    subgraph "Alert & Notification"
        ALERT_MGR[Alert Manager]
        TELEGRAM[Telegram Bot]
        SLACK[Slack Integration]
        SMS[SMS Notifications]
        WEBSOCKET[WebSocket Updates]
    end
    
    %% Monitoring & Analytics
    subgraph "Monitoring & Analytics"
        GRAFANA[Grafana Dashboards]
        PROMETHEUS[Prometheus Metrics]
        ANALYTICS[Risk Analytics]
        HOTSPOTS[Hotspot Analysis]
    end
    
    %% API Layer
    subgraph "API & Web Interface"
        FASTAPI[FastAPI Server]
        REST[REST APIs]
        DOCS[API Documentation]
        WEB_UI[Web Dashboard]
    end
    
    %% Background Processing
    subgraph "Background Tasks"
        CELERY[Celery Workers]
        BEAT[Celery Beat]
        CLEANUP[Cleanup Tasks]
    end
    
    %% Connections
    RTSP --> LIVE_PROC
    UPLOAD --> BATCH_PROC
    LIVE --> REALTIME_PROC
    
    LIVE_PROC --> DETECT
    BATCH_PROC --> DETECT
    REALTIME_PROC --> DETECT
    
    DETECT --> TRACK
    TRACK --> GENDER
    TRACK --> POSE
    TRACK --> VIOLENCE
    
    GENDER --> RULES
    POSE --> RULES
    VIOLENCE --> RULES
    RULES --> CONTEXT
    CONTEXT --> TEMPORAL
    
    TEMPORAL --> ALERT_MGR
    ALERT_MGR --> TELEGRAM
    ALERT_MGR --> SLACK
    ALERT_MGR --> SMS
    ALERT_MGR --> WEBSOCKET
    
    ALERT_MGR --> FACE_BLUR
    FACE_BLUR --> MINIO
    ALERT_MGR --> MINIO
    
    MINIO --> RETENTION
    RETENTION --> ENCRYPT
    
    ALERT_MGR --> POSTGRES
    ALERT_MGR --> TIMESCALE
    ALERT_MGR --> REDIS
    
    POSTGRES --> ANALYTICS
    TIMESCALE --> HOTSPOTS
    REDIS --> CELERY
    
    FASTAPI --> REST
    FASTAPI --> DOCS
    FASTAPI --> WEB_UI
    
    REST --> GRAFANA
    REST --> PROMETHEUS
    
    CELERY --> BEAT
    BEAT --> CLEANUP
    CLEANUP --> MINIO
    
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef aiStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef riskStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef privacyStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef storageStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef alertStyle fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef monitorStyle fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef apiStyle fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef bgStyle fill:#f5f5f5,stroke:#616161,stroke-width:1px
    
    class RTSP,UPLOAD,LIVE inputStyle
    class DETECT,TRACK,GENDER,POSE,VIOLENCE aiStyle
    class RULES,CONTEXT,TEMPORAL riskStyle
    class FACE_BLUR,RETENTION,ENCRYPT privacyStyle
    class MINIO,POSTGRES,TIMESCALE,REDIS storageStyle
    class ALERT_MGR,TELEGRAM,SLACK,SMS,WEBSOCKET alertStyle
    class GRAFANA,PROMETHEUS,ANALYTICS,HOTSPOTS monitorStyle
    class FASTAPI,REST,DOCS,WEB_UI apiStyle
    class LIVE_PROC,BATCH_PROC,REALTIME_PROC,CELERY,BEAT,CLEANUP bgStyle
"""
    
    return diagram

def generate_data_flow_diagram():
    """Generate data flow diagram"""
    
    diagram = """
graph LR
    %% Data Flow
    A[Video Input] --> B[Frame Extraction]
    B --> C[Person Detection]
    C --> D[Gender Classification]
    C --> E[Pose Estimation]
    C --> F[Violence Detection]
    
    D --> G[Risk Analysis]
    E --> G
    F --> G
    
    G --> H{High Risk?}
    H -->|Yes| I[Generate Alert]
    H -->|No| J[Continue Monitoring]
    
    I --> K[Face Blurring]
    K --> L[Store Clip]
    L --> M[Send Notifications]
    
    J --> N[Update Analytics]
    N --> O[Store Metrics]
    
    %% Styling
    classDef processStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef decisionStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef alertStyle fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    classDef storageStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class A,B,C,D,E,F processStyle
    class H decisionStyle
    class I,K,L,M alertStyle
    class J,N,O storageStyle
"""
    
    return diagram

def generate_system_components_diagram():
    """Generate system components diagram"""
    
    diagram = """
graph TB
    subgraph "Frontend Layer"
        WEB[Web Dashboard]
        API_DOCS[API Documentation]
    end
    
    subgraph "API Gateway"
        FASTAPI[FastAPI Server<br/>Port 8000]
    end
    
    subgraph "Core Services"
        DETECTOR[Person Detector]
        TRACKER[Person Tracker]
        GENDER_CLF[Gender Classifier]
        POSE_EST[Pose Estimator]
        VIOLENCE_CLF[Violence Classifier]
        RISK_ANALYZER[Risk Analyzer]
    end
    
    subgraph "Processing Services"
        LIVE_PROC[Live Processor]
        BATCH_PROC[Batch Processor]
        VIDEO_PROC[Video Processor]
    end
    
    subgraph "Background Services"
        CELERY_WORKER[Celery Worker]
        CELERY_BEAT[Celery Beat]
        CLEANUP[Cleanup Service]
    end
    
    subgraph "Storage Services"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis)]
        MINIO[MinIO]
    end
    
    subgraph "Monitoring Services"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
    end
    
    %% Connections
    WEB --> FASTAPI
    API_DOCS --> FASTAPI
    
    FASTAPI --> DETECTOR
    FASTAPI --> TRACKER
    FASTAPI --> GENDER_CLF
    FASTAPI --> POSE_EST
    FASTAPI --> VIOLENCE_CLF
    FASTAPI --> RISK_ANALYZER
    
    LIVE_PROC --> DETECTOR
    BATCH_PROC --> DETECTOR
    VIDEO_PROC --> DETECTOR
    
    DETECTOR --> TRACKER
    TRACKER --> GENDER_CLF
    TRACKER --> POSE_EST
    TRACKER --> VIOLENCE_CLF
    
    GENDER_CLF --> RISK_ANALYZER
    POSE_EST --> RISK_ANALYZER
    VIOLENCE_CLF --> RISK_ANALYZER
    
    RISK_ANALYZER --> POSTGRES
    RISK_ANALYZER --> REDIS
    RISK_ANALYZER --> MINIO
    
    CELERY_WORKER --> POSTGRES
    CELERY_WORKER --> REDIS
    CELERY_BEAT --> CELERY_WORKER
    CLEANUP --> MINIO
    
    PROMETHEUS --> FASTAPI
    GRAFANA --> PROMETHEUS
    
    %% Styling
    classDef frontendStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef apiStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef coreStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef processStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef bgStyle fill:#f5f5f5,stroke:#616161,stroke-width:2px
    classDef storageStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef monitorStyle fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    
    class WEB,API_DOCS frontendStyle
    class FASTAPI apiStyle
    class DETECTOR,TRACKER,GENDER_CLF,POSE_EST,VIOLENCE_CLF,RISK_ANALYZER coreStyle
    class LIVE_PROC,BATCH_PROC,VIDEO_PROC processStyle
    class CELERY_WORKER,CELERY_BEAT,CLEANUP bgStyle
    class POSTGRES,REDIS,MINIO storageStyle
    class PROMETHEUS,GRAFANA monitorStyle
"""
    
    return diagram

def main():
    """Generate all architecture diagrams"""
    
    print("=== Safety Detection System Architecture Diagrams ===\n")
    
    print("1. MAIN ARCHITECTURE DIAGRAM:")
    print("=" * 50)
    print(generate_architecture_diagram())
    print("\n")
    
    print("2. DATA FLOW DIAGRAM:")
    print("=" * 50)
    print(generate_data_flow_diagram())
    print("\n")
    
    print("3. SYSTEM COMPONENTS DIAGRAM:")
    print("=" * 50)
    print(generate_system_components_diagram())
    print("\n")
    
    print("=== USAGE INSTRUCTIONS ===")
    print("1. Copy any diagram code above")
    print("2. Paste into Mermaid Live Editor: https://mermaid.live/")
    print("3. Or use in GitHub/GitLab markdown with ```mermaid code blocks")
    print("4. Or save as .mmd files and render with Mermaid CLI")

if __name__ == "__main__":
    main()
