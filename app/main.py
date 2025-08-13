"""
Moondream2 VLM API Server
High-performance image description generation using Moondream2 Vision Language Model
Based on official Hugging Face documentation: https://huggingface.co/vikhyatk/moondream2
"""

import os
import io
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import psutil

# Import transformers for Moondream2
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Model will be loaded on first request.")

# Configure logging
logger.add("logs/moondream2.log", rotation="10 MB", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Moondream2 VLM API",
    description="High-performance image description generation using Moondream2 Vision Language Model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
device = None

# Pydantic models
class ImageDescriptionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(..., description="Text prompt for image description")
    max_tokens: int = Field(default=512, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")

class ImageDescriptionResponse(BaseModel):
    description: str
    confidence: float
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, Any]  # Allow nested structures and empty dicts

# Initialize model function
def initialize_model():
    """Initialize Moondream2 model using transformers"""
    global model, tokenizer, device
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available. Skipping model initialization.")
        return False
    
    try:
        logger.info("Initializing Moondream2 model using transformers...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer from Hugging Face
        logger.info("Loading Moondream2 model from Hugging Face...")
        
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map="auto"  # Automatically handle device placement
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", 
            revision="2025-06-21", 
            trust_remote_code=True
        )
        
        logger.info("Model initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

# Utility functions
def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }

def get_gpu_memory() -> Optional[Dict[str, Any]]:
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        }
    return {"error": "No GPU available"}  # Better than empty dict

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize model during startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Moondream2 VLM API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_memory = get_gpu_memory()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage={
            "system": get_memory_usage(),
            "gpu": gpu_memory if gpu_memory else {}
        }
    )

@app.post("/describe", response_model=ImageDescriptionResponse)
async def describe_image(request: ImageDescriptionRequest):
    """Generate image description using Moondream2"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Generate description using the correct Moondream2 API
        with torch.no_grad():
            # Use the query method for visual question answering
            response = model.query(image, request.prompt)
            answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        # Calculate confidence (placeholder - Moondream2 doesn't provide confidence scores)
        confidence = 0.85  # Default confidence
        
        return ImageDescriptionResponse(
            description=answer,
            confidence=confidence,
            processing_time=processing_time,
            model_info={
                "model": "Moondream2",
                "device": str(device),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/describe/file")
async def describe_image_file(
    file: UploadFile = File(...),
    prompt: str = "Describe this image in detail.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate image description from uploaded file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate description using the correct Moondream2 API
        import time
        start_time = time.time()
        
        with torch.no_grad():
            response = model.query(image, prompt)
            answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        return {
            "description": answer,
            "confidence": 0.85,
            "processing_time": processing_time,
            "model_info": {
                "model": "Moondream2",
                "device": str(device),
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/caption")
async def caption_image(
    file: UploadFile = File(...),
    length: str = "normal"  # "short" or "normal"
):
    """Generate image caption using Moondream2"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate caption using the correct Moondream2 API
        import time
        start_time = time.time()
        
        with torch.no_grad():
            response = model.caption(image, length=length)
            caption = response["caption"]
        
        processing_time = time.time() - start_time
        
        return {
            "caption": caption,
            "length": length,
            "processing_time": processing_time,
            "model_info": {
                "model": "Moondream2",
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    object_name: str = "person"
):
    """Detect objects in image using Moondream2"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect objects using the correct Moondream2 API
        import time
        start_time = time.time()
        
        with torch.no_grad():
            response = model.detect(image, object_name)
            objects = response["objects"]
        
        processing_time = time.time() - start_time
        
        return {
            "objects": objects,
            "object_name": object_name,
            "count": len(objects),
            "processing_time": processing_time,
            "model_info": {
                "model": "Moondream2",
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "Moondream2",
        "model_type": "Vision Language Model",
        "model_id": "vikhyatk/moondream2",
        "revision": "2025-06-21",
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": get_memory_usage(),
        "gpu_memory": get_gpu_memory()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
