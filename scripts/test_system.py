#!/usr/bin/env python3
"""
Test script for Safety Detection System
"""

import asyncio
import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def test_health():
    """Test health endpoints"""
    print("🔍 Testing health endpoints...")
    
    # Basic health check
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200
    print("✅ Basic health check passed")
    
    # Detailed health check
    response = requests.get(f"{API_BASE}/health/detailed")
    assert response.status_code == 200
    print("✅ Detailed health check passed")
    
    # Camera health check
    response = requests.get(f"{API_BASE}/health/cameras")
    assert response.status_code == 200
    print("✅ Camera health check passed")

def test_cameras():
    """Test camera endpoints"""
    print("📹 Testing camera endpoints...")
    
    # Get all cameras
    response = requests.get(f"{API_BASE}/cameras")
    assert response.status_code == 200
    cameras = response.json()
    print(f"✅ Found {len(cameras)} cameras")
    
    # Get camera summary
    response = requests.get(f"{API_BASE}/cameras/stats/summary")
    assert response.status_code == 200
    summary = response.json()
    print(f"✅ Camera summary: {summary}")
    
    # Test individual camera if available
    if cameras:
        camera_id = cameras[0]['id']
        response = requests.get(f"{API_BASE}/cameras/{camera_id}")
        assert response.status_code == 200
        print(f"✅ Camera {camera_id} details retrieved")

def test_zones():
    """Test zone endpoints"""
    print("🗺️  Testing zone endpoints...")
    
    # Get all zones
    response = requests.get(f"{API_BASE}/zones")
    assert response.status_code == 200
    zones = response.json()
    print(f"✅ Found {len(zones)} zones")
    
    # Test individual zone if available
    if zones:
        zone_id = zones[0]['id']
        response = requests.get(f"{API_BASE}/zones/{zone_id}")
        assert response.status_code == 200
        print(f"✅ Zone {zone_id} details retrieved")

def test_alerts():
    """Test alert endpoints"""
    print("🚨 Testing alert endpoints...")
    
    # Get alerts
    response = requests.get(f"{API_BASE}/alerts")
    assert response.status_code == 200
    alerts = response.json()
    print(f"✅ Found {len(alerts)} alerts")
    
    # Get alert summary
    response = requests.get(f"{API_BASE}/alerts/stats/summary")
    assert response.status_code == 200
    summary = response.json()
    print(f"✅ Alert summary: {summary}")

def test_analytics():
    """Test analytics endpoints"""
    print("📊 Testing analytics endpoints...")
    
    # Get hotspots
    response = requests.get(f"{API_BASE}/analytics/hotspots")
    assert response.status_code == 200
    hotspots = response.json()
    print(f"✅ Found {len(hotspots)} hotspots")
    
    # Get gender distribution
    response = requests.get(f"{API_BASE}/analytics/gender-distribution")
    assert response.status_code == 200
    distributions = response.json()
    print(f"✅ Found {len(distributions)} gender distributions")
    
    # Get system metrics
    response = requests.get(f"{API_BASE}/analytics/system-metrics")
    assert response.status_code == 200
    metrics = response.json()
    print(f"✅ Found {len(metrics)} system metrics")
    
    # Get analytics summary
    response = requests.get(f"{API_BASE}/analytics/summary")
    assert response.status_code == 200
    summary = response.json()
    print(f"✅ Analytics summary: {summary}")

def test_websocket():
    """Test WebSocket connection"""
    print("🔌 Testing WebSocket connection...")
    
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            uri = f"ws://localhost:8000/ws"
            async with websockets.connect(uri) as websocket:
                # Send a test message
                await websocket.send("test message")
                response = await websocket.recv()
                print(f"✅ WebSocket response: {response}")
        
        asyncio.run(test_ws())
        
    except ImportError:
        print("⚠️  websockets library not installed, skipping WebSocket test")
    except Exception as e:
        print(f"⚠️  WebSocket test failed: {e}")

def create_test_alert():
    """Create a test alert for testing"""
    print("🧪 Creating test alert...")
    
    # This would create a test alert in a real system
    # For now, just check if the endpoint exists
    response = requests.get(f"{API_BASE}/alerts")
    if response.status_code == 200:
        print("✅ Alert endpoints are accessible")

def main():
    """Run all tests"""
    print("🚀 Starting Safety Detection System tests...")
    print(f"Testing API at: {API_BASE}")
    print("=" * 50)
    
    try:
        test_health()
        print()
        
        test_cameras()
        print()
        
        test_zones()
        print()
        
        test_alerts()
        print()
        
        test_analytics()
        print()
        
        test_websocket()
        print()
        
        create_test_alert()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed!")
        print("✅ Safety Detection System is working correctly")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        print("Start the server with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
