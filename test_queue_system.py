#!/usr/bin/env python3
"""
Test script for Moondream2 Queue System
Demonstrates motion detection, queue management, and webhook functionality
"""

import asyncio
import base64
import json
import time
from PIL import Image
import numpy as np
import requests
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
WEBHOOK_URL = "http://localhost:5000/webhook"  # Simulated webhook endpoint

def create_test_image(width: int = 640, height: int = 480) -> str:
    """Create a test image and return base64 encoded string"""
    # Create a simple test image
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = image.tobytes()
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64

def simulate_motion_detection(camera_id: str, motion_intensity: float = 0.5) -> Dict[str, Any]:
    """Simulate motion detection request"""
    frame_data = create_test_image()
    
    payload = {
        "camera_id": camera_id,
        "frame_data": frame_data,
        "motion_intensity": motion_intensity,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location": f"Camera {camera_id} Location",
        "metadata": {
            "zone": "test_zone",
            "sensitivity": "medium",
            "test_run": True
        }
    }
    
    return payload

def test_motion_detection():
    """Test basic motion detection endpoint"""
    print("🔍 Testing Motion Detection Endpoint...")
    
    # Test different motion intensities
    test_cases = [
        ("camera_001", 0.3, "Low motion"),
        ("camera_002", 0.6, "Medium motion"),
        ("camera_003", 0.9, "High motion"),
    ]
    
    job_ids = []
    
    for camera_id, intensity, description in test_cases:
        print(f"  📹 {description} (Camera: {camera_id}, Intensity: {intensity})")
        
        payload = simulate_motion_detection(camera_id, intensity)
        
        try:
            response = requests.post(f"{API_BASE_URL}/motion/detect", json=payload)
            response.raise_for_status()
            
            result = response.json()
            job_ids.append(result["job_id"])
            
            print(f"    ✅ Job created: {result['job_id']}")
            print(f"    📊 Status: {result['status']}")
            print(f"    📍 Queue position: {result['queue_position']}")
            print(f"    ⏱️  Estimated wait: {result['estimated_wait_time']:.1f}s")
            
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Error: {e}")
    
    return job_ids

def test_motion_detection_with_webhook():
    """Test motion detection with webhook"""
    print("\n🔗 Testing Motion Detection with Webhook...")
    
    payload = simulate_motion_detection("camera_webhook", 0.8)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/motion/detect-with-webhook",
            params={
                "camera_id": payload["camera_id"],
                "frame_data": payload["frame_data"],
                "motion_intensity": payload["motion_intensity"],
                "webhook_url": WEBHOOK_URL,
                "webhook_headers": json.dumps({"Content-Type": "application/json"}),
                "prompt": "Describe any suspicious activity in this surveillance frame",
                "metadata": json.dumps(payload["metadata"])
            }
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"  ✅ Webhook job created: {result['job_id']}")
        print(f"  📊 Status: {result['status']}")
        print(f"  🔗 Webhook configured: {result['webhook_configured']}")
        
        return result["job_id"]
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")
        return None

def monitor_jobs(job_ids: list, duration: int = 30):
    """Monitor job status for a specified duration"""
    print(f"\n📊 Monitoring {len(job_ids)} jobs for {duration} seconds...")
    
    start_time = time.time()
    completed_jobs = set()
    
    while time.time() - start_time < duration and len(completed_jobs) < len(job_ids):
        for job_id in job_ids:
            if job_id in completed_jobs:
                continue
                
            try:
                response = requests.get(f"{API_BASE_URL}/queue/job/{job_id}")
                response.raise_for_status()
                
                job_status = response.json()
                status = job_status["status"]
                
                if status in ["completed", "failed", "cancelled"]:
                    completed_jobs.add(job_id)
                    
                    if status == "completed":
                        caption = job_status.get("result", {}).get("caption", "No caption")
                        processing_time = job_status.get("result", {}).get("processing_time", 0)
                        print(f"  ✅ Job {job_id[:8]}... completed in {processing_time:.2f}s")
                        print(f"     📝 Caption: {caption[:100]}...")
                    else:
                        print(f"  ❌ Job {job_id[:8]}... {status}")
                else:
                    position = job_status.get("queue_position", "N/A")
                    wait_time = job_status.get("estimated_wait_time", 0)
                    print(f"  ⏳ Job {job_id[:8]}... {status} (pos: {position}, wait: {wait_time:.1f}s)")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error monitoring job {job_id}: {e}")
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"\n📈 Monitoring completed. {len(completed_jobs)}/{len(job_ids)} jobs finished.")

def test_queue_stats():
    """Test queue statistics endpoint"""
    print("\n📊 Testing Queue Statistics...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/queue/stats")
        response.raise_for_status()
        
        stats = response.json()
        
        print(f"  📈 Total jobs: {stats['total_jobs']}")
        print(f"  🔄 Active jobs: {stats['active_jobs']}")
        print(f"  📋 Queue size: {stats['queue_size']}")
        print(f"  ⚙️  Max workers: {stats['max_concurrent_jobs']}")
        print(f"  🏥 System status: {stats['system_status']}")
        
        if stats.get('average_processing_time'):
            print(f"  ⏱️  Avg processing time: {stats['average_processing_time']:.2f}s")
        
        if stats.get('webhook_success_rate'):
            print(f"  🔗 Webhook success rate: {stats['webhook_success_rate']:.1%}")
        
        print("  📊 Status breakdown:")
        for status, count in stats['status_counts'].items():
            print(f"    - {status}: {count}")
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")

def test_health_check():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        health = response.json()
        
        print(f"  🏥 Status: {health['status']}")
        print(f"  🤖 Model loaded: {health['model_loaded']}")
        print(f"  🎮 GPU available: {health['gpu_available']}")
        
        if health.get('memory_usage'):
            system_mem = health['memory_usage'].get('system', {})
            gpu_mem = health['memory_usage'].get('gpu', {})
            
            if system_mem:
                print(f"  💾 System memory: {system_mem.get('rss_mb', 0):.1f}MB")
            
            if gpu_mem and 'allocated_mb' in gpu_mem:
                print(f"  🎮 GPU memory: {gpu_mem['allocated_mb']:.1f}MB / {gpu_mem['total_mb']:.1f}MB")
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("🚀 Moondream2 Queue System Test")
    print("=" * 50)
    
    # Test health check first
    test_health_check()
    
    # Test motion detection
    job_ids = test_motion_detection()
    
    # Test webhook functionality
    webhook_job_id = test_motion_detection_with_webhook()
    if webhook_job_id:
        job_ids.append(webhook_job_id)
    
    # Monitor jobs
    if job_ids:
        monitor_jobs(job_ids, duration=60)  # Monitor for 60 seconds
    
    # Test queue statistics
    test_queue_stats()
    
    print("\n✅ Test completed!")
    print("\n💡 Tips:")
    print("  - Check the API documentation at http://localhost:8000/docs")
    print("  - Monitor logs in logs/moondream2.log")
    print("  - Use /queue/stats to monitor system performance")
    print("  - Implement a webhook receiver to get real-time notifications")

if __name__ == "__main__":
    main()
