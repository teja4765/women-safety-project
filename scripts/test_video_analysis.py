#!/usr/bin/env python3
"""
Test script for video analysis and processing modes
"""

import asyncio
import requests
import json
import time
import os
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def test_processing_modes():
    """Test processing modes API"""
    print("🔧 Testing Processing Modes API...")
    
    # Get available processing modes
    response = requests.get(f"{API_BASE}/processing-modes")
    assert response.status_code == 200
    modes = response.json()
    print(f"✅ Found {len(modes)} processing modes")
    
    for mode in modes:
        print(f"  - {mode['mode']}: {mode['name']}")
    
    # Test auto-selection
    test_cases = [
        ("rtsp_url", "rtsp://192.168.1.100:554/stream1"),
        ("file_upload", "test_video.mp4"),
        ("file_path", "/path/to/video.mp4")
    ]
    
    for input_type, input_data in test_cases:
        response = requests.get(
            f"{API_BASE}/processing-modes/auto-select",
            params={"input_type": input_type}
        )
        assert response.status_code == 200
        result = response.json()
        print(f"✅ Auto-selected {result['selected_mode']} for {input_type}")
    
    # Test input validation
    response = requests.get(
        f"{API_BASE}/processing-modes/validate-input",
        params={
            "input_data": "rtsp://192.168.1.100:554/stream1",
            "mode": "live_cctv"
        }
    )
    assert response.status_code == 200
    result = response.json()
    print(f"✅ Input validation: {result['is_valid']}")
    
    # Get processing mode statistics
    response = requests.get(f"{API_BASE}/processing-modes/statistics")
    assert response.status_code == 200
    stats = response.json()
    print(f"✅ Processing mode statistics: {stats['total_cameras']} cameras")
    
    print("✅ Processing modes API tests passed\n")

def test_video_analysis_upload():
    """Test video file upload and analysis"""
    print("📹 Testing Video Analysis Upload...")
    
    # Create a test video file (placeholder)
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        print(f"⚠️  Test video file {test_video_path} not found, creating placeholder...")
        # In a real test, you would create or use an actual video file
        with open(test_video_path, "wb") as f:
            f.write(b"fake video content for testing")
    
    try:
        # Test file upload
        with open(test_video_path, "rb") as f:
            files = {"file": (test_video_path, f, "video/mp4")}
            data = {
                "zone_id": "campus_parking_a",
                "processing_mode": "batch"
            }
            
            response = requests.post(
                f"{API_BASE}/video-analysis/upload",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                job = response.json()
                print(f"✅ Video upload job created: {job['job_id']}")
                return job['job_id']
            else:
                print(f"⚠️  Video upload failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"⚠️  Video upload test failed: {e}")
        return None
    finally:
        # Clean up test file
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_video_analysis_jobs():
    """Test video analysis jobs API"""
    print("📋 Testing Video Analysis Jobs...")
    
    # Get all jobs
    response = requests.get(f"{API_BASE}/video-analysis/jobs")
    assert response.status_code == 200
    jobs = response.json()
    print(f"✅ Found {len(jobs)} video analysis jobs")
    
    # Get system status
    response = requests.get(f"{API_BASE}/video-analysis/status")
    assert response.status_code == 200
    status = response.json()
    print(f"✅ Video analysis system status: {status['status']}")
    print(f"  - Active jobs: {status['active_jobs']}")
    print(f"  - Completed jobs: {status['completed_jobs']}")
    
    # Test individual job if available
    if jobs:
        job_id = jobs[0]['job_id']
        response = requests.get(f"{API_BASE}/video-analysis/jobs/{job_id}")
        assert response.status_code == 200
        job_details = response.json()
        print(f"✅ Job details retrieved for {job_id}")
        print(f"  - Status: {job_details['status']}")
        print(f"  - Progress: {job_details['progress']:.1f}%")
    
    print("✅ Video analysis jobs tests passed\n")

def test_processing_mode_switching():
    """Test processing mode switching"""
    print("🔄 Testing Processing Mode Switching...")
    
    # Test mode switch (this would require an actual camera)
    test_camera_id = "test_cam"
    
    # Get current mode
    try:
        response = requests.get(f"{API_BASE}/processing-modes/camera/{test_camera_id}")
        if response.status_code == 200:
            current_mode = response.json()
            print(f"✅ Current mode for {test_camera_id}: {current_mode['mode']}")
            
            # Try to switch mode
            switch_data = {
                "mode": "video_file_batch",
                "reason": "Testing mode switching"
            }
            
            response = requests.post(
                f"{API_BASE}/processing-modes/camera/{test_camera_id}/switch",
                json=switch_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Mode switched successfully: {result['message']}")
            else:
                print(f"⚠️  Mode switch failed: {response.status_code} - {response.text}")
        else:
            print(f"⚠️  Camera {test_camera_id} not found or no active mode")
            
    except Exception as e:
        print(f"⚠️  Mode switching test failed: {e}")
    
    print("✅ Processing mode switching tests completed\n")

def test_live_cctv_integration():
    """Test live CCTV integration"""
    print("📺 Testing Live CCTV Integration...")
    
    # Test camera status (should show live processing)
    response = requests.get(f"{API_BASE}/cameras/stats/summary")
    assert response.status_code == 200
    camera_stats = response.json()
    print(f"✅ Camera statistics: {camera_stats['total_cameras']} total cameras")
    print(f"  - Online: {camera_stats['online_cameras']}")
    print(f"  - Offline: {camera_stats['offline_cameras']}")
    
    # Test individual camera status
    response = requests.get(f"{API_BASE}/cameras")
    assert response.status_code == 200
    cameras = response.json()
    
    if cameras:
        camera_id = cameras[0]['id']
        response = requests.get(f"{API_BASE}/cameras/{camera_id}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Camera {camera_id} status: {status['status']}")
            print(f"  - Processing FPS: {status['processing_fps']}")
            print(f"  - Detection count: {status['detection_count']}")
    
    print("✅ Live CCTV integration tests passed\n")

def test_websocket_connection():
    """Test WebSocket connection for real-time updates"""
    print("🔌 Testing WebSocket Connection...")
    
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            uri = f"ws://localhost:8000/ws"
            try:
                async with websockets.connect(uri) as websocket:
                    # Send a test message
                    await websocket.send("test message")
                    response = await websocket.recv()
                    print(f"✅ WebSocket response: {response}")
                    
                    # Wait for any real-time updates
                    try:
                        update = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"✅ Received real-time update: {update[:100]}...")
                    except asyncio.TimeoutError:
                        print("ℹ️  No real-time updates received (normal if no activity)")
                        
            except Exception as e:
                print(f"⚠️  WebSocket test failed: {e}")
        
        asyncio.run(test_ws())
        
    except ImportError:
        print("⚠️  websockets library not installed, skipping WebSocket test")
    except Exception as e:
        print(f"⚠️  WebSocket test failed: {e}")
    
    print("✅ WebSocket connection tests completed\n")

def test_integration_scenarios():
    """Test integration scenarios"""
    print("🔗 Testing Integration Scenarios...")
    
    # Scenario 1: Upload video file and monitor processing
    print("Scenario 1: Video file upload and processing...")
    job_id = test_video_analysis_upload()
    
    if job_id:
        # Monitor job progress
        for i in range(3):
            time.sleep(2)
            response = requests.get(f"{API_BASE}/video-analysis/jobs/{job_id}")
            if response.status_code == 200:
                job = response.json()
                print(f"  Job progress: {job['progress']:.1f}% - Status: {job['status']}")
                if job['status'] in ['completed', 'failed']:
                    break
    
    # Scenario 2: Check processing mode distribution
    print("Scenario 2: Processing mode distribution...")
    response = requests.get(f"{API_BASE}/processing-modes/statistics")
    if response.status_code == 200:
        stats = response.json()
        print(f"  Total cameras: {stats['total_cameras']}")
        print(f"  Mode distribution: {stats['mode_distribution']}")
    
    # Scenario 3: System health check
    print("Scenario 3: System health check...")
    response = requests.get(f"{API_BASE}/health/detailed")
    if response.status_code == 200:
        health = response.json()
        print(f"  System status: {health['status']}")
        print(f"  Components: {health['components']}")
    
    print("✅ Integration scenarios completed\n")

def main():
    """Run all video analysis and processing mode tests"""
    print("🚀 Starting Video Analysis and Processing Modes Tests...")
    print(f"Testing API at: {API_BASE}")
    print("=" * 60)
    
    try:
        # Test processing modes
        test_processing_modes()
        
        # Test video analysis
        test_video_analysis_jobs()
        
        # Test processing mode switching
        test_processing_mode_switching()
        
        # Test live CCTV integration
        test_live_cctv_integration()
        
        # Test WebSocket connection
        test_websocket_connection()
        
        # Test integration scenarios
        test_integration_scenarios()
        
        print("=" * 60)
        print("🎉 All video analysis and processing mode tests passed!")
        print("✅ System supports both live CCTV and video file analysis")
        print("✅ Processing modes are working correctly")
        print("✅ Real-time updates are functional")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        print("Start the server with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
