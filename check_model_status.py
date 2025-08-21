#!/usr/bin/env python3
"""
Model Status Checker for Moondream2 VLM API
Provides user-friendly model validation and status information
"""

import os
import sys
import requests
import json
from pathlib import Path

def check_local_model():
    """Check local model file status"""
    print("🔍 Checking local model file...")
    
    model_path = "/app/models/moondream2-onnx/moondream-0_5b-int8.mf"
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Model file exists: {model_path}")
        print(f"📊 File size: {size:,} bytes ({size / (1024*1024):.1f} MB)")
        
        if size > 0:
            print("✅ Model file appears valid")
            return True
        else:
            print("❌ Model file is empty")
            return False
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

def check_api_status(base_url="http://localhost:8000"):
    """Check API status and model validation"""
    print(f"\n🌐 Checking API status at {base_url}...")
    
    try:
        # Check if API is running
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            health_data = response.json()
            print(f"📊 Model loaded: {health_data.get('model_loaded', 'Unknown')}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API - is it running?")
        return False
    except Exception as e:
        print(f"❌ API check error: {e}")
        return False
    
    try:
        # Check model validation
        response = requests.get(f"{base_url}/model/validate", timeout=5)
        if response.status_code == 200:
            validation_data = response.json()
            is_valid = validation_data.get('is_valid', False)
            
            if is_valid:
                print("✅ Model validation passed")
                model_info = validation_data.get('validation_info', {}).get('model_info', {})
                if model_info:
                    print(f"📊 Model size: {model_info.get('size_bytes', 0):,} bytes")
            else:
                print("❌ Model validation failed")
                error_msg = validation_data.get('validation_info', {}).get('error_message', 'Unknown error')
                print(f"💬 Error: {error_msg}")
                
                suggestions = validation_data.get('validation_info', {}).get('suggestions', [])
                if suggestions:
                    print("💡 Suggestions:")
                    for suggestion in suggestions:
                        print(f"   - {suggestion}")
            
            return is_valid
        else:
            print(f"❌ Model validation check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model validation check error: {e}")
        return False

def main():
    """Main function"""
    print("🌙 Moondream2 VLM Model Status Checker")
    print("=" * 50)
    
    # Check if running inside container
    if os.path.exists("/app"):
        print("🐳 Running inside Docker container")
        local_ok = check_local_model()
    else:
        print("💻 Running on host system")
        local_ok = False
    
    # Check API status
    api_ok = check_api_status()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"   Local model file: {'✅ OK' if local_ok else '❌ Missing/Invalid'}")
    print(f"   API status: {'✅ OK' if api_ok else '❌ Failed'}")
    
    if local_ok and api_ok:
        print("\n🎉 Everything looks good! The model is ready to use.")
    elif not local_ok and not api_ok:
        print("\n🔧 Setup needed:")
        print("   1. Start the Docker container: docker-compose up --build")
        print("   2. Wait for model download to complete")
        print("   3. Check status again")
    elif local_ok and not api_ok:
        print("\n🔧 API issue detected:")
        print("   1. Check if the container is running: docker ps")
        print("   2. Check container logs: docker-compose logs moondream2-vlm")
        print("   3. Restart if needed: docker-compose restart")
    else:
        print("\n🔧 Model issue detected:")
        print("   1. The model file may be missing or corrupted")
        print("   2. Restart the container to re-download: docker-compose down && docker-compose up --build")

if __name__ == "__main__":
    main()
