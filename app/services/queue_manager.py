"""
Simple Queue Manager for Caption Generation
Handles basic caption generation requests with webhook delivery
"""
import asyncio
import logging
import uuid
import time
from typing import Dict, Optional
from datetime import datetime
import aiohttp

from app.models.schemas import (
    CaptionRequest,
    SurveillanceRequest,
    CaptionResult,
    JobStatus,
    WebhookPayload,
    JobResponse,
    JobStatusResponse,
    QueueStats
)

logger = logging.getLogger(__name__)

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
            await self.processing_queue.put((job_id, request, "caption"))
            
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

    async def add_surveillance_job(
        self, 
        timestamp: datetime,
        image_data: str,
        alias: str,
        eventId: str,
        webhook_url: Optional[str] = None
    ) -> str:
        """Add a surveillance caption generation job to the queue"""
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create request
            request = SurveillanceRequest(
                timestamp=timestamp,
                image_data=image_data,
                alias=alias,
                eventId=eventId,
                webhook_url=webhook_url
            )
            
            # Check if queue is full
            if self.processing_queue.qsize() >= self.max_queue_size:
                logger.warning(f"Queue full ({self.max_queue_size}), rejecting new job")
                raise Exception("Queue is full, please try again later")
            
            # Add to queue
            await self.processing_queue.put((job_id, request, "surveillance"))
            
            # Create job response
            job_response = JobStatusResponse(
                job_id=job_id,
                status=JobStatus.QUEUED,
                message=f"Surveillance caption job queued for processing (Event: {eventId})",
                created_at=datetime.now()
            )
            
            self.jobs[job_id] = job_response
            
            logger.info(f"Surveillance job {job_id} added to queue (Event: {eventId}, Alias: {alias})")
            return job_id
            
        except Exception as e:
            logger.error(f"Error adding surveillance job: {e}")
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
                    job_data = await asyncio.wait_for(
                        self.processing_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Handle different job types
                if len(job_data) == 3:
                    job_id, request, job_type = job_data
                    if job_type == "surveillance":
                        task = asyncio.create_task(self._process_surveillance_job(job_id, request))
                    else:
                        task = asyncio.create_task(self._process_job(job_id, request))
                else:
                    # Legacy format for backward compatibility
                    job_id, request = job_data
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
            
            # Process with Moondream2 (this will be implemented in main.py)
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
            
            # Send webhook if configured
            if request.webhook_url:
                await self._send_webhook(job_id, request.webhook_url, result, None)
            
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
            
            # Send webhook with error if configured
            if request.webhook_url:
                await self._send_webhook(job_id, request.webhook_url, None, error_msg)
            
            logger.error(f"Job {job_id} failed: {e}")
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    async def _process_surveillance_job(self, job_id: str, request: SurveillanceRequest):
        """Process a surveillance caption generation job"""
        start_time = time.time()
        
        try:
            # Update status to processing
            await self._update_job_status(
                job_id, 
                JobStatus.PROCESSING, 
                f"Processing surveillance frame with Moondream2 (Event: {request.eventId})"
            )
            
            # Generate caption using Moondream2's predefined method
            result = await self._generate_surveillance_caption(request)
            
            # Update status to completed
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            await self._update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                f"Surveillance caption completed (Event: {request.eventId})",
                result=result
            )
            
            # Send webhook if configured
            if request.webhook_url:
                await self._send_webhook(job_id, request.webhook_url, result, None)
            
            logger.info(f"Surveillance job {job_id} completed in {processing_time:.2f}s (Event: {request.eventId})")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Surveillance caption generation failed: {str(e)}"
            
            await self._update_job_status(
                job_id, 
                JobStatus.FAILED, 
                error_msg,
                error_message=error_msg
            )
            
            # Send webhook with error if configured
            if request.webhook_url:
                await self._send_webhook(job_id, request.webhook_url, None, error_msg)
            
            logger.error(f"Surveillance job {job_id} failed: {e}")
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _generate_caption(self, request: CaptionRequest) -> CaptionResult:
        """Generate caption using Moondream2 model"""
        # This will be implemented in main.py with the actual model
        # For now, return a placeholder
        return CaptionResult(
            caption="[Placeholder: Moondream2 caption generation will be implemented]",
            processing_time=0.0,
            eventId="",
            alias="",
            timestamp=datetime.now()
        )

    async def _generate_surveillance_caption(self, request: SurveillanceRequest) -> CaptionResult:
        """Generate surveillance caption using Moondream2 model"""
        # This will be implemented in main.py with the actual model
        # For now, return a placeholder
        return CaptionResult(
            caption="[Placeholder: Moondream2 surveillance caption will be implemented]",
            processing_time=0.0,
            eventId=request.eventId,
            alias=request.alias,
            timestamp=request.timestamp
        )
    
    async def _send_webhook(
        self, 
        job_id: str, 
        webhook_url: str, 
        result: Optional[CaptionResult], 
        error_message: Optional[str]
    ):
        """Send webhook notification"""
        try:
            webhook_payload = WebhookPayload(
                job_id=job_id,
                status=JobStatus.COMPLETED if result else JobStatus.FAILED,
                result=result,
                error_message=error_message,
                timestamp=datetime.now()
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=webhook_payload.dict(),
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully for job {job_id}")
                    else:
                        logger.warning(f"Webhook failed for job {job_id}: {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending webhook for job {job_id}: {e}")
    
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
