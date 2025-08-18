#!/usr/bin/env python3
"""
Simple test script for Moondream2 Queue System
Demonstrates basic caption generation with queue and webhook functionality
"""

import base64
import time
from PIL import Image
import numpy as np
import requests
from typing import Dict, Any
import io

# Configuration
API_BASE_URL = "http://localhost:8000"
WEBHOOK_URL = "http://localhost:5000/webhook"  # Simulated webhook endpoint

def create_test_image(width: int = 640, height: int = 480) -> str:
    """Create a test image and return base64 encoded string"""
    # Create a simple test image
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_base64

def test_caption_generation():
    """Test basic caption generation endpoint"""
    print("🔍 Testing Caption Generation Endpoint...")
    
    # Create test image
    image_data = create_test_image()
    
    # Test without webhook
    print("  📝 Testing without webhook...")
    try:
        response = requests.post(f"{API_BASE_URL}/caption/generate", json={
            "image_data": image_data,
            "prompt": "Describe what you see in this image"
        })
        response.raise_for_status()
        
        result = response.json()
        job_id = result["job_id"]
        
        print(f"    ✅ Job created: {job_id}")
        print(f"    📊 Status: {result['status']}")
        
        return job_id
        
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Error: {e}")
        return None

def test_caption_generation_with_webhook():
    """Test caption generation with webhook"""
    print("\n🔗 Testing Caption Generation with Webhook...")
    
    # Create test image
    image_data = create_test_image()
    
    try:
        response = requests.post(f"{API_BASE_URL}/caption/generate", json={
            "image_data": image_data,
            "webhook_url": WEBHOOK_URL,
            "prompt": "Describe any objects or people in this image"
        })
        response.raise_for_status()
        
        result = response.json()
        job_id = result["job_id"]
        
        print(f"  ✅ Webhook job created: {job_id}")
        print(f"  📊 Status: {result['status']}")
        
        return job_id
        
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
                response = requests.get(f"{API_BASE_URL}/caption/job/{job_id}")
                response.raise_for_status()
                
                job_status = response.json()
                status = job_status["status"]
                
                if status in ["completed", "failed"]:
                    completed_jobs.add(job_id)
                    
                    if status == "completed":
                        caption = job_status.get("result", {}).get("caption", "No caption")
                        processing_time = job_status.get("result", {}).get("processing_time", 0)
                        print(f"  ✅ Job {job_id[:8]}... completed in {processing_time:.2f}s")
                        print(f"     📝 Caption: {caption[:100]}...")
                    else:
                        error_msg = job_status.get("error_message", "Unknown error")
                        print(f"  ❌ Job {job_id[:8]}... failed: {error_msg}")
                else:
                    print(f"  ⏳ Job {job_id[:8]}... {status}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error monitoring job {job_id}: {e}")
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"\n📈 Monitoring completed. {len(completed_jobs)}/{len(job_ids)} jobs finished.")

def test_queue_stats():
    """Test queue statistics endpoint"""
    print("\n📊 Testing Queue Statistics...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/caption/stats")
        response.raise_for_status()
        
        stats = response.json()
        
        print(f"  📈 Total jobs: {stats['total_jobs']}")
        print(f"  🔄 Active jobs: {stats['active_jobs']}")
        print(f"  📋 Queue size: {stats['queue_size']}")
        
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
    print("🚀 Moondream2 Simple Queue System Test")
    print("=" * 50)
    
    # Test health check first
    test_health_check()
    
    # Test caption generation
    job_ids = []
    
    # Test without webhook
    job_id = test_caption_generation()
    if job_id:
        job_ids.append(job_id)
    
    # Test with webhook
    webhook_job_id = test_caption_generation_with_webhook()
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
    print("  - Use /caption/stats to monitor queue performance")
    print("  - Implement a webhook receiver to get real-time notifications")

if __name__ == "__main__":
    main()
