#!/bin/bash

# Safety Detection System Setup Script

set -e

echo "🚀 Setting up Safety Detection System..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/clips
mkdir -p data/models
mkdir -p logs
mkdir -p static
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources

# Set permissions
chmod 755 data/clips
chmod 755 data/models
chmod 755 logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Database
DATABASE_URL=postgresql://safety_user:safety_pass@localhost:5432/safety_db

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=safety-clips
MINIO_SECURE=false

# Notifications (optional)
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# TELEGRAM_CHAT_ID=your_telegram_chat_id
# SLACK_WEBHOOK_URL=your_slack_webhook_url

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Application
DEBUG=false
ALLOWED_ORIGINS=["*"]
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Download YOLO model if not present
if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading YOLO model..."
    mkdir -p models
    # This would download the model in a real setup
    echo "⚠️  Please download yolov8n.pt to models/ directory"
fi

# Create monitoring configuration
echo "📊 Setting up monitoring..."

# Prometheus configuration
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'safety-detection'
    static_configs:
      - targets: ['safety-detection:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your configuration"
echo "2. Download YOLO model: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt"
echo "3. Start the system: docker-compose up -d"
echo "4. Access the API: http://localhost:8000/docs"
echo "5. Access Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "For development:"
echo "- Run with: uvicorn app.main:app --reload"
echo "- Test with: python scripts/test_system.py"
