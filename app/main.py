"""
Moondream2 VLM API Server
High-performance image description generation using Moondream2 Vision Language Model
Based on official Hugging Face documentation: https://huggingface.co/vikhyatk/moondream2
"""

import os
import io
import base64
import time
import tempfile
import threading
import queue as thread_queue
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import psutil
import requests

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

# In-process queue and job management
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_STARTED = "started"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

class EnqueueJsonPayload(BaseModel):
    event_id: str
    cam_alias: str
    utc_timestamp: str
    image_base64: Optional[str] = None
    question: Optional[str] = None

class JobInfo(BaseModel):
    job_id: str
    event_id: str
    status: str
    caption: Optional[str] = None
    answer: Optional[str] = None
    question: Optional[str] = None
    caption_flag: bool = False
    error: Optional[str] = None
    enqueued_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None

# Thread-safe structures
_queue: "thread_queue.Queue[Dict[str, Any]]" = thread_queue.Queue()
_jobs: Dict[str, JobInfo] = {}
_event_to_job: Dict[str, str] = {}
_jobs_lock = threading.Lock()
_counters = {
    "enqueued": 0,
    "started": 0,
    "completed": 0,
    "failed": 0,
}
_workers: List[threading.Thread] = []
_shutdown_flag = threading.Event()
_cleanup_shutdown_flag = threading.Event()
_cleanup_thread: Optional[threading.Thread] = None

RESULT_WEBHOOK_URL = os.getenv("RESULT_WEBHOOK_URL", "")
QUEUE_WORKERS = int(os.getenv("QUEUE_WORKERS", "1"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "60"))

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

# Webhook posting helper
def post_status(event_id: str, status: str) -> None:
    if not RESULT_WEBHOOK_URL:
        return
    payload = {"event_id": event_id, "status": status}
    for attempt in range(3):
        try:
            resp = requests.post(RESULT_WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code < 500:
                return
        except Exception as exc:
            if attempt == 2:
                logger.warning(f"Webhook post failed after retries for event_id={event_id}: {exc}")
        time.sleep(0.5)

# Unified job processor
def process_job(job_payload: Dict[str, Any]) -> None:
    job_id: str = job_payload["job_id"]
    event_id: str = job_payload["event_id"]
    image_path: str = job_payload["image_path"]
    question: Optional[str] = job_payload.get("question") or ""
    caption_flag: bool = False
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job.status = JOB_STATUS_STARTED
                job.started_at = time.time()
                _counters["started"] += 1
        post_status(event_id, JOB_STATUS_STARTED)

        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run model
        if model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        # Always compute caption
        with torch.no_grad():
            cap_resp = model.caption(image, length="normal")
            caption_text = cap_resp.get("caption") if isinstance(cap_resp, dict) else str(cap_resp)
        answer_text: Optional[str] = None
        # If question present, also compute answer
        if question.strip():
            with torch.no_grad():
                ans_resp = model.query(image, question)
                answer_text = ans_resp.get("answer") if isinstance(ans_resp, dict) else str(ans_resp)
        caption_flag = True
        duration_ms = (time.time() - start) * 1000.0

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job.status = JOB_STATUS_COMPLETED
                job.finished_at = time.time()
                job.duration_ms = duration_ms
                job.caption = caption_text
                job.answer = answer_text
                job.question = question
                job.caption_flag = caption_flag
                _counters["completed"] += 1
        # Log final job response payload
        try:
            logger.info(
                f"Job completed | job_id={job_id} event_id={event_id} duration_ms={duration_ms:.2f} "
                f"caption_flag={caption_flag} question='{(question or '').strip()}' "
                f"caption='{(caption_text or '').strip()[:200]}' "
                f"answer='{answer_text}'"
            )
        except Exception:
            pass
        post_status(event_id, JOB_STATUS_COMPLETED)
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job.status = JOB_STATUS_FAILED
                job.finished_at = time.time()
                job.error = str(e)
                _counters["failed"] += 1
        post_status(event_id, JOB_STATUS_FAILED)
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

# Worker loop
def worker_loop(worker_idx: int) -> None:
    logger.info(f"Queue worker {worker_idx} started")
    while not _shutdown_flag.is_set():
        try:
            job_payload = _queue.get(timeout=0.5)
        except thread_queue.Empty:
            continue
        try:
            process_job(job_payload)
        finally:
            _queue.task_done()
    logger.info(f"Queue worker {worker_idx} stopped")

def start_workers():
    global _workers
    if _workers:
        return
    for i in range(max(1, QUEUE_WORKERS)):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        _workers.append(t)

def stop_workers():
    _shutdown_flag.set()
    for t in _workers:
        t.join(timeout=2.0)

def cleanup_loop():
    logger.info("Job cleanup thread started")
    while not _cleanup_shutdown_flag.is_set():
        now = time.time()
        expired_ids: List[str] = []
        with _jobs_lock:
            for job_id, job in list(_jobs.items()):
                if job.finished_at:
                    if now - job.finished_at >= JOB_TTL_SECONDS:
                        expired_ids.append(job_id)
            for job_id in expired_ids:
                job = _jobs.pop(job_id, None)
                # remove event mapping if it points to this job
                if job:
                    mapped = _event_to_job.get(job.event_id)
                    if mapped == job_id:
                        _event_to_job.pop(job.event_id, None)
        _cleanup_shutdown_flag.wait(timeout=5.0)
    logger.info("Job cleanup thread stopped")

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize model during startup")
    start_workers()
    # Start cleanup thread
    global _cleanup_thread
    if _cleanup_thread is None:
        _cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        _cleanup_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully stop worker threads on shutdown"""
    try:
        stop_workers()
    except Exception:
        pass
    try:
        _cleanup_shutdown_flag.set()
        if _cleanup_thread:
            _cleanup_thread.join(timeout=2.0)
    except Exception:
        pass

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

# Queue endpoints
@app.post("/queue/enqueue")
async def enqueue_job(
    request: Request,
    file: Optional[UploadFile] = File(None),
    event_id: Optional[str] = Form(None),
    cam_alias: Optional[str] = Form(None),
    utc_timestamp: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    json_payload: Optional[dict] = None
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            json_payload = await request.json()
    except Exception:
        json_payload = None

    if json_payload is None and file is None:
        raise HTTPException(status_code=400, detail="Provide either multipart form or JSON body")

    if json_payload is not None:
        try:
            payload = EnqueueJsonPayload(**json_payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
        event_id_val = payload.event_id
        cam_alias_val = payload.cam_alias
        utc_ts_val = payload.utc_timestamp
        question_val = payload.question or ""
        b64 = payload.image_base64
        if not b64:
            raise HTTPException(status_code=400, detail="image_base64 is required in JSON mode")
        try:
            b64_str = b64
            if b64_str.startswith('data:image'):
                b64_str = b64_str.split(',')[1]
            img_bytes = base64.b64decode(b64_str)
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    else:
        if not event_id or not cam_alias or not utc_timestamp:
            raise HTTPException(status_code=400, detail="event_id, cam_alias, utc_timestamp are required")
        if file is None:
            raise HTTPException(status_code=400, detail="file is required in multipart mode")
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="file must be an image")
        event_id_val = event_id
        cam_alias_val = cam_alias
        utc_ts_val = utc_timestamp
        question_val = (question or "")
        try:
            data = await file.read()
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    with _jobs_lock:
        existing_job_id = _event_to_job.get(event_id_val)
        if existing_job_id:
            job = _jobs.get(existing_job_id)
            if job:
                return {"job_id": existing_job_id, "idempotent": True}

    job_id = str(uuid.uuid4())
    job_info = JobInfo(
        job_id=job_id,
        event_id=event_id_val,
        status=JOB_STATUS_QUEUED,
        caption=None,
        question=question_val,
        caption_flag=False,
        error=None,
        enqueued_at=time.time(),
        started_at=None,
        finished_at=None,
        duration_ms=None,
    )
    with _jobs_lock:
        _jobs[job_id] = job_info
        _event_to_job[event_id_val] = job_id
        _counters["enqueued"] += 1
    post_status(event_id_val, JOB_STATUS_QUEUED)

    _queue.put({
        "job_id": job_id,
        "event_id": event_id_val,
        "image_path": tmp_path,
        "question": question_val,
        "cam_alias": cam_alias_val,
        "utc_timestamp": utc_ts_val,
    })

    return {"job_id": job_id, "idempotent": False}

@app.get("/queue/stats")
async def queue_stats():
    with _jobs_lock:
        return {
            "queue_size": _queue.qsize(),
            "enqueued": _counters["enqueued"],
            "started": _counters["started"],
            "completed": _counters["completed"],
            "failed": _counters["failed"],
            "workers": len(_workers),
            "jobs_total": len(_jobs),
        }

@app.get("/queue/job/{job_id}")
async def queue_job_detail(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump()

@app.post("/test/process")
async def test_process(
    request: Request,
    file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),  # Changed from question_form to question
):
    # Synchronous processing endpoint for testing (no queue)
    json_payload: Optional[dict] = None
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            json_payload = await request.json()
    except Exception:
        json_payload = None

    tmp_path = None
    question_val = ""
    if json_payload is not None:
        image_base64 = json_payload.get("image_base64")
        question_val = (json_payload.get("question") or "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required for JSON mode")
        try:
            b64_str = image_base64
            if b64_str.startswith('data:image'):
                b64_str = b64_str.split(',')[1]
            img_bytes = base64.b64decode(b64_str)
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="file is required in multipart mode")
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="file must be an image")
        question_val = (question or "")  # Changed from question_form to question
        logger.info(f"Received question: '{question_val}'")  # Add debugging
        try:
            data = await file.read()
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run synchronously using the same logic as the worker
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        image = Image.open(tmp_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        start = time.time()
        # Always compute caption
        with torch.no_grad():
            cap_resp = model.caption(image, length="normal")
            caption_text = cap_resp.get("caption") if isinstance(cap_resp, dict) else str(cap_resp)
        answer_text: Optional[str] = None
        # If question present, also compute answer
        if question_val.strip():
            logger.info(f"Processing question: '{question_val}'")  # Add debugging
            with torch.no_grad():
                ans_resp = model.query(image, question_val)
                answer_text = ans_resp.get("answer") if isinstance(ans_resp, dict) else str(ans_resp)
                logger.info(f"Model response: {ans_resp}")  # Add debugging
        else:
            logger.info("No question provided, skipping answer generation")
        caption_flag = True
        duration_ms = (time.time() - start) * 1000.0
        return {
            "caption": caption_text,
            "answer": answer_text,
            "question": question_val,
            "caption_flag": caption_flag,
            "duration_ms": duration_ms,
        }
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

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
