"""
Model file checking utilities for Moondream2
Validates model file existence and integrity
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ModelChecker:
    """Utility class for checking model file status"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.model_filename = os.path.basename(model_path)
    
    def check_model_file(self) -> Tuple[bool, str]:
        """
        Check if the model file exists and is valid
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                return False, f"Model directory does not exist: {self.model_dir}"
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                return False, f"Model file does not exist: {self.model_path}"
            
            # Check if it's a file (not a directory)
            if not os.path.isfile(self.model_path):
                return False, f"Model path is not a file: {self.model_path}"
            
            # Check file size (should be > 0)
            file_size = os.path.getsize(self.model_path)
            if file_size == 0:
                return False, f"Model file is empty: {self.model_path}"
            
            # Check if file is readable
            if not os.access(self.model_path, os.R_OK):
                return False, f"Model file is not readable: {self.model_path}"
            
            # Log success
            logger.info(f"Model file validation successful: {self.model_path} ({file_size} bytes)")
            return True, f"Model file is valid ({file_size} bytes)"
            
        except Exception as e:
            error_msg = f"Error checking model file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_model_info(self) -> dict:
        """Get detailed information about the model file"""
        info = {
            "model_path": self.model_path,
            "model_dir": self.model_dir,
            "model_filename": self.model_filename,
            "exists": False,
            "size_bytes": 0,
            "readable": False,
            "error": None
        }
        
        try:
            if os.path.exists(self.model_path):
                info["exists"] = True
                info["size_bytes"] = os.path.getsize(self.model_path)
                info["readable"] = os.access(self.model_path, os.R_OK)
            else:
                info["error"] = "File does not exist"
                
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    def suggest_fixes(self) -> list:
        """Suggest fixes for common model file issues"""
        suggestions = []
        
        if not os.path.exists(self.model_dir):
            suggestions.append(f"Create model directory: mkdir -p {self.model_dir}")
        
        if not os.path.exists(self.model_path):
            suggestions.append(f"Download model file to: {self.model_path}")
            suggestions.append("Run the container with proper model download setup")
        
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) == 0:
            suggestions.append(f"Model file is empty, re-download: {self.model_path}")
        
        if os.path.exists(self.model_path) and not os.access(self.model_path, os.R_OK):
            suggestions.append(f"Fix file permissions: chmod 644 {self.model_path}")
        
        return suggestions

def validate_model_setup(model_path: str) -> Tuple[bool, dict]:
    """
    Comprehensive model validation
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple[bool, dict]: (is_valid, validation_info)
    """
    checker = ModelChecker(model_path)
    is_valid, error_msg = checker.check_model_file()
    
    validation_info = {
        "is_valid": is_valid,
        "error_message": error_msg if not is_valid else None,
        "model_info": checker.get_model_info(),
        "suggestions": checker.suggest_fixes() if not is_valid else []
    }
    
    return is_valid, validation_info
