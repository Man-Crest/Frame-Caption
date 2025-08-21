#!/usr/bin/env python3
"""
Test script for model validation functionality
"""

import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_model_validation():
    """Test the model validation functionality"""
    
    print("🧪 Testing Model Validation...")
    
    try:
        from app.utils.model_checker import validate_model_setup, ModelChecker
        
        # Test with the expected model path
        model_path = "/app/models/moondream2-onnx/moondream-0_5b-int8.mf"
        
        print(f"🔍 Testing model path: {model_path}")
        
        # Test validation
        is_valid, validation_info = validate_model_setup(model_path)
        
        print(f"✅ Validation result: {is_valid}")
        print(f"📊 Model info: {validation_info['model_info']}")
        
        if not is_valid:
            print(f"❌ Error: {validation_info['error_message']}")
            print("💡 Suggestions:")
            for suggestion in validation_info.get('suggestions', []):
                print(f"   - {suggestion}")
        
        # Test ModelChecker class directly
        print("\n🔍 Testing ModelChecker class...")
        checker = ModelChecker(model_path)
        checker_is_valid, checker_error = checker.check_model_file()
        
        print(f"✅ ModelChecker result: {checker_is_valid}")
        if not checker_is_valid:
            print(f"❌ ModelChecker error: {checker_error}")
        
        # Test with a non-existent path
        print("\n🔍 Testing with non-existent path...")
        fake_path = "/fake/path/model.mf"
        fake_is_valid, fake_validation = validate_model_setup(fake_path)
        
        print(f"✅ Fake path validation (should be False): {fake_is_valid}")
        if not fake_is_valid:
            print(f"❌ Expected error: {fake_validation['error_message']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_validation()
    if success:
        print("\n✅ All model validation tests completed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
