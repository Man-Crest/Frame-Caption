"""
Moondream2 VLM API Server with Simple Queue Management
High-performance image description generation using Moondream2 Vision Language Model
Enhanced with simple queue management for caption generation
"""

import os
import io
import base64
import time
import uuid
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum

from PIL import Image
import moondream as md
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import psutil

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
device = "cpu"  # Fixed to CPU for 0.5B ONNX model
BACKEND = os.getenv("MOONDREAM_BACKEND", "mf").lower()
MF_MODEL_PATH = os.getenv("MOONDREAM_MF_PATH", "/app/models/moondream2-onnx/moondream-0_5b-int8.mf")

# Simple queue management
class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CaptionRequest(BaseModel):
    """Request model for caption generation"""
    image_data: str = Field(..., description="Base64 encoded image")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for result delivery")
    prompt: str = Field(default="Describe what you see in this image", description="Caption prompt")

class CaptionResult(BaseModel):
    """Result model for caption generation"""
    caption: str = Field(..., description="Generated image caption")
    processing_time: float = Field(..., description="Processing time in seconds")

class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation timestamp")

class JobStatusResponse(BaseModel):
    """Response model for job status check"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    result: Optional[CaptionResult] = Field(None, description="Caption result")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class QueueStats(BaseModel):
    """Queue statistics model"""
    total_jobs: int = Field(..., description="Total number of jobs")
    active_jobs: int = Field(..., description="Currently active jobs")
    queue_size: int = Field(..., description="Jobs waiting in queue")
    status_counts: Dict[str, int] = Field(..., description="Job counts by status")

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

# Simple Queue Manager (inline to avoid import issues)
class SimpleQueueManager:
    """Simple queue manager for caption generation"""
    
    def __init__(self, max_workers: int = 2, max_queue_size: int = 50):
        # Queue management
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: Dict[str, JobStatusResponse] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self.max_concurrent_jobs = max_workers
        self.max_queue_size = max_queue_size
        
        # Worker task
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
        
    async def start(self):
        """Start the queue manager and worker"""
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._worker_loop())
            logger.info(f"Queue manager started with {self.max_concurrent_jobs} workers")
    
    async def stop(self):
        """Stop the queue manager"""
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Queue manager stopped")
    
    async def add_caption_job(
        self, 
        image_data: str,
        webhook_url: Optional[str] = None,
        prompt: str = "Describe what you see in this image"
    ) -> str:
        """Add a caption generation job to the queue"""
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create request
            request = CaptionRequest(
                image_data=image_data,
                webhook_url=webhook_url,
                prompt=prompt
            )
            
            # Check if queue is full
            if self.processing_queue.qsize() >= self.max_queue_size:
                logger.warning(f"Queue full ({self.max_queue_size}), rejecting new job")
                raise Exception("Queue is full, please try again later")
            
            # Add to queue
            await self.processing_queue.put((job_id, request))
            
            # Create job response
            job_response = JobStatusResponse(
                job_id=job_id,
                status=JobStatus.QUEUED,
                message="Caption generation job queued for processing",
                created_at=datetime.now()
            )
            
            self.jobs[job_id] = job_response
            
            logger.info(f"Caption job {job_id} added to queue")
            return job_id
            
        except Exception as e:
            logger.error(f"Error adding caption job: {e}")
            raise
    
    async def _worker_loop(self):
        """Main worker loop for processing jobs"""
        while self.is_running:
            try:
                # Wait for available worker slot
                while len(self.active_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(0.1)
                
                # Get next job from queue
                try:
                    job_id, request = await asyncio.wait_for(
                        self.processing_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Create task for processing
                task = asyncio.create_task(self._process_job(job_id, request))
                self.active_jobs[job_id] = task
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_job(self, job_id: str, request: CaptionRequest):
        """Process a single caption generation job"""
        start_time = time.time()
        
        try:
            # Update status to processing
            await self._update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                "Processing image with Moondream2"
            )
            
            # Process with Moondream2
            result = await self._generate_caption(request)
            
            # Update status to completed
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            await self._update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                "Caption generation completed",
                result=result
            )
            
            logger.info(f"Job {job_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Caption generation failed: {str(e)}"
            
            await self._update_job_status(
                job_id, 
                JobStatus.FAILED, 
                error_msg,
                error_message=error_msg
            )
            
            logger.error(f"Job {job_id} failed: {e}")
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    async def _generate_caption(self, request: CaptionRequest) -> CaptionResult:
        """Generate caption using Moondream2 model"""
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
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """Get the status of a specific job"""
        return self.jobs.get(job_id)
    
    async def _update_job_status(
        self, 
        job_id: str, 
        status: JobStatus, 
        message: str = "",
        result: Optional[CaptionResult] = None,
        error_message: Optional[str] = None
    ):
        """Update the status of a job"""
        try:
            if job_id in self.jobs:
                self.jobs[job_id].status = status
                self.jobs[job_id].message = message
                
                if result:
                    self.jobs[job_id].result = result
                
                if error_message:
                    self.jobs[job_id].error_message = error_message
                
                if status == JobStatus.COMPLETED:
                    self.jobs[job_id].completed_at = datetime.now()
                
                logger.debug(f"Job {job_id} status updated to {status.value}")
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    async def get_queue_stats(self) -> QueueStats:
        """Get queue statistics"""
        try:
            total_jobs = len(self.jobs)
            status_counts = {}
            
            for job in self.jobs.values():
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return QueueStats(
                total_jobs=total_jobs,
                active_jobs=len(self.active_jobs),
                queue_size=self.processing_queue.qsize(),
                status_counts=status_counts
            )
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            raise

# Global queue manager instance
queue_manager = SimpleQueueManager()

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



# Initialize model function
def initialize_model():
    """Initialize Moondream2 model using the working simple approach (CPU-only)"""
    global model, tokenizer, device

    # Fixed to CPU for 0.5B ONNX model
    device = "cpu"
    logger.info(f"Using device: {device}")

    # Simple model file check
    if not os.path.exists(MF_MODEL_PATH):
        logger.error(f"Model file not found: {MF_MODEL_PATH}")
        return False
    
    # Check file size (should be several MB)
    file_size = os.path.getsize(MF_MODEL_PATH)
    if file_size < 1000000:  # Less than 1MB
        logger.error(f"Model file seems too small: {file_size} bytes")
        return False
    
    logger.info(f"Model file found: {MF_MODEL_PATH} ({file_size} bytes)")

    try:
        logger.info("Initializing Moondream2 0.5B model (CPU-only)...")
        # Use the working simple approach
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
    """Health check endpoint (CPU-only)"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=False,  # Fixed to False for CPU-only
        memory_usage={
            "system": get_memory_usage(),
            "gpu": {"error": "CPU-only deployment"}
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

@app.post("/caption/direct")
async def caption_image_direct(
    file: UploadFile = File(...),
    prompt: str = "Describe what you see in this image"
):
    """Direct caption generation without queue - for VLM testing"""
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
            "prompt": prompt,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating direct caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct caption generation failed: {str(e)}")

@app.post("/caption/direct/base64")
async def caption_image_direct_base64(request: CaptionRequest):
    """Direct caption generation from base64 image - for VLM testing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        
        # Generate caption using the correct Moondream2 API pattern
        start_time = time.time()
        
        encoded_image = model.encode_image(image)
        response = model.caption(encoded_image)
        caption = response["caption"]
        
        processing_time = time.time() - start_time
        
        meta = _current_model_meta()
        return {
            "caption": caption,
            "prompt": request.prompt,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating direct caption from base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct caption generation failed: {str(e)}")

@app.post("/query/direct")
async def query_image_direct(
    file: UploadFile = File(...),
    question: str = "What do you see in this image?"
):
    """Direct image query without queue - for VLM testing"""
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
        
        # Generate answer using the correct Moondream2 API pattern
        start_time = time.time()
        
        encoded_image = model.encode_image(image)
        response = model.query(encoded_image, question)
        answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        meta = _current_model_meta()
        return {
            "answer": answer,
            "question": question,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating direct query answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct query generation failed: {str(e)}")

@app.post("/query/direct/base64")
async def query_image_direct_base64(
    image_data: str,
    question: str = "What do you see in this image?"
):
    """Direct image query from base64 - for VLM testing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image = decode_base64_image(image_data)
        
        # Generate answer using the correct Moondream2 API pattern
        start_time = time.time()
        
        encoded_image = model.encode_image(image)
        response = model.query(encoded_image, question)
        answer = response["answer"]
        
        processing_time = time.time() - start_time
        
        meta = _current_model_meta()
        return {
            "answer": answer,
            "question": question,
            "processing_time": processing_time,
            "model_info": {
                "model": meta["model_name"],
                "model_id": meta["model_id"],
                "revision": meta["revision"],
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating direct query answer from base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct query generation failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information (CPU-only)"""
    meta = _current_model_meta()
    
    # Simple model validation
    is_valid = os.path.exists(MF_MODEL_PATH) and os.path.getsize(MF_MODEL_PATH) > 1000000
    file_size = os.path.getsize(MF_MODEL_PATH) if os.path.exists(MF_MODEL_PATH) else 0
    
    return {
        "model_name": meta["model_name"],
        "model_type": meta["model_type"],
        "model_id": meta["model_id"],
        "revision": meta["revision"],
        "device": device,  # Will be "cpu"
        "gpu_available": False,  # Fixed to False
        "memory_usage": get_memory_usage(),
        "gpu_memory": {"error": "CPU-only deployment"},
        "queue_enabled": True,
        "queue_workers": queue_manager.max_concurrent_jobs,
        "model_loaded": model is not None,
        "model_validation": {
            "is_valid": is_valid,
            "model_path": MF_MODEL_PATH,
            "file_size": file_size,
            "exists": os.path.exists(MF_MODEL_PATH)
        }
    }

@app.get("/model/validate")
async def validate_model():
    """Validate model file and return detailed status"""
    is_valid = os.path.exists(MF_MODEL_PATH) and os.path.getsize(MF_MODEL_PATH) > 1000000
    file_size = os.path.getsize(MF_MODEL_PATH) if os.path.exists(MF_MODEL_PATH) else 0
    
    return {
        "is_valid": is_valid,
        "model_path": MF_MODEL_PATH,
        "file_size": file_size,
        "exists": os.path.exists(MF_MODEL_PATH),
        "model_loaded": model is not None,
        "device": device  # Will be "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
