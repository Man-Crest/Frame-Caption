"""
Moondream2 VLM API Server with Simple Queue Management
High-performance image description generation using Moondream2 Vision Language Model
Enhanced with simple queue management for caption generation
"""

import os
import io
import base64
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import moondream as md
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import psutil

# We run ONNX-only for the 0.5B path. Remove transformers usage.
TRANSFORMERS_AVAILABLE = False

# Import queue management
from app.services.queue_manager import queue_manager
from app.models.schemas import (
    CaptionRequest,
    CaptionResult,
    JobStatus,
    JobResponse,
    JobStatusResponse,
    QueueStats
)

# Configure logging
logger.add("logs/moondream2.log", rotation="10 MB", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Moondream2 VLM API with Simple Queue Management",
    description="High-performance image description generation with simple queue management",
    version="2.0.0",
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
BACKEND = os.getenv("MOONDREAM_BACKEND", "mf").lower()
MF_MODEL_PATH = os.getenv("MOONDREAM_MF_PATH", "/app/models/moondream2-onnx/moondream-0_5b-int8.mf")

def _current_model_meta() -> Dict[str, Any]:
    if BACKEND == "onnx":
        return {
            "model_name": "Moondream2-0.5B-ONNX",
            "model_type": "Vision Language Model (ONNX)",
            "model_id": "vikhyatk/moondream2",
            "revision": "onnx",
        }
    else:
        return {
            "model_name": "Moondream2",
            "model_type": "Vision Language Model",
            "model_id": "vikhyatk/moondream2",
            "revision": "2025-06-21",
        }

# Pydantic models (keeping existing ones for backward compatibility)
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
    memory_usage: Dict[str, Any]

# Initialize model function
def initialize_model():
    """Initialize Moondream2 model using the correct API pattern"""
    global model, tokenizer, device

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info("Initializing Moondream2 0.5B model...")
        # Use the correct API pattern as shown in the example
        model = md.vl(model=MF_MODEL_PATH)
        tokenizer = None
        logger.info("Moondream2 model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Moondream2 model initialization failed: {e}")
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
    try:
        import torch
    except Exception:
        torch = None
    if torch and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        }
    return {"error": "No GPU available"}

# Moondream2 caption generation function for queue
async def generate_caption_with_moondream2(request: CaptionRequest) -> CaptionResult:
    """Generate caption using Moondream2 model for queue system"""
    if model is None:
        raise Exception("Model not loaded")
    
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        
        # Generate caption using the correct Moondream2 API pattern
        # First encode the image, then generate caption
        encoded_image = model.encode_image(image)
        response = model.caption(encoded_image)
        caption = response["caption"]
        
        processing_time = time.time() - start_time
        
        return CaptionResult(
            caption=caption,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating caption with Moondream2: {str(e)}")
        raise

# Override the placeholder method in queue manager
queue_manager._generate_caption = generate_caption_with_moondream2

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize model and queue manager on startup"""
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize model during startup")
    
    # Start queue manager
    await queue_manager.start()
    logger.info("Queue manager started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop queue manager on shutdown"""
    await queue_manager.stop()
    logger.info("Queue manager stopped")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Moondream2 VLM API with Simple Queue Management",
        "version": "2.0.0",
        "docs": "/docs",
        "queue_status": "active"
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

# Simple Queue Management Endpoints

@app.post("/caption/generate", response_model=JobResponse)
async def generate_caption(request: CaptionRequest):
    """Generate caption for an image - adds job to processing queue"""
    try:
        # Add job to queue
        job_id = await queue_manager.add_caption_job(
            image_data=request.image_data,
            webhook_url=request.webhook_url,
            prompt=request.prompt
        )
        
        # Get job status
        job_status = await queue_manager.get_job_status(job_id)
        
        return JobResponse(
            job_id=job_id,
            status=job_status.status,
            message=job_status.message,
            created_at=job_status.created_at
        )
        
    except Exception as e:
        logger.error(f"Error processing caption generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")

@app.get("/caption/job/{job_id}", response_model=JobStatusResponse)
async def get_caption_job_status(job_id: str):
    """Get the status of a specific caption generation job"""
    try:
        job_status = await queue_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/caption/stats", response_model=QueueStats)
async def get_caption_queue_stats():
    """Get caption generation queue statistics"""
    try:
        stats = await queue_manager.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

# Existing endpoints (keeping for backward compatibility)

@app.post("/describe", response_model=ImageDescriptionResponse)
async def describe_image(request: ImageDescriptionRequest):
    """Generate image description using Moondream2 (synchronous)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Generate description using the correct Moondream2 API pattern
        # First encode the image, then query
        encoded_image = model.encode_image(image)
        response = model.query(encoded_image, request.prompt)
        answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        # Calculate confidence (placeholder - Moondream2 doesn't provide confidence scores)
        confidence = 0.85  # Default confidence
        
        meta = _current_model_meta()
        return ImageDescriptionResponse(
            description=answer,
            confidence=confidence,
            processing_time=processing_time,
            model_info={
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
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
        
        # Generate description using the correct Moondream2 API pattern
        start_time = time.time()
        
        encoded_image = model.encode_image(image)
        response = model.query(encoded_image, prompt)
        answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        meta = _current_model_meta()
        return {
            "description": answer,
            "confidence": 0.85,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
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
        
        # Generate caption using the correct Moondream2 API pattern
        start_time = time.time()
        
        encoded_image = model.encode_image(image)
        response = model.caption(encoded_image)
        caption = response["caption"]
        
        processing_time = time.time() - start_time
        
        meta = _current_model_meta()
        return {
            "caption": caption,
            "length": length,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    meta = _current_model_meta()
    return {
        "model_name": meta["model_name"],
        "model_type": meta["model_type"],
        "model_id": meta["model_id"],
        "revision": meta["revision"],
        "device": str(device),
        "gpu_available": False,
        "memory_usage": get_memory_usage(),
        "gpu_memory": get_gpu_memory(),
        "queue_enabled": True,
        "queue_workers": queue_manager.max_concurrent_jobs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
