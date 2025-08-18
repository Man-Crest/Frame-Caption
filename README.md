# Moondream2 VLM API with Queue Management

High-performance image description generation using Moondream2 Vision Language Model, enhanced with queue management for surveillance systems.

## Features

### Core Features
- **Moondream2 Integration**: State-of-the-art vision language model for image understanding
- **Queue Management**: Asynchronous processing with priority queuing
- **Motion Detection API**: Dedicated endpoints for surveillance systems
- **Webhook Support**: Real-time notifications for processing results
- **Priority Queuing**: Intelligent job prioritization based on motion intensity
- **System Monitoring**: Comprehensive statistics and health monitoring

### Surveillance-Specific Features
- **Motion Detection Endpoints**: `/motion/detect` and `/motion/detect-with-webhook`
- **Queue Management**: `/queue/*` endpoints for job monitoring
- **Priority Levels**: Critical, High, Normal, Low based on motion intensity
- **Webhook Delivery**: Automatic result delivery to external systems
- **Job Tracking**: Real-time status updates and queue position monitoring

## Architecture

```
Motion Detection → API Endpoint → Priority Queue → Worker Pool → Moondream2 → Webhook Response
```

### Queue System
- **Priority Queue**: Jobs processed based on motion intensity
- **Worker Pool**: Configurable concurrent processing (default: 2 workers)
- **Queue Limits**: Configurable queue size with overflow protection
- **Job Lifecycle**: Queued → Processing → Completed/Failed/Cancelled

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Moondream2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the server**:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Motion Detection

#### POST `/motion/detect`
Add a motion detection job to the processing queue.

**Request Body**:
```json
{
  "camera_id": "camera_001",
  "frame_data": "base64_encoded_image",
  "motion_intensity": 0.7,
  "timestamp": "2024-01-01T12:00:00Z",
  "location": "Front Door",
  "metadata": {
    "zone": "entrance",
    "sensitivity": "high"
  }
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Motion detection job queued for processing",
  "queue_position": 3,
  "estimated_wait_time": 6.5,
  "webhook_configured": false
}
```

#### POST `/motion/detect-with-webhook`
Add a motion detection job with webhook notification.

**Parameters**:
- `camera_id`: Camera identifier
- `frame_data`: Base64 encoded image
- `motion_intensity`: Motion intensity (0.0-1.0)
- `webhook_url`: URL for result delivery
- `webhook_headers`: Optional headers for webhook
- `prompt`: Custom caption prompt
- `metadata`: Additional metadata

### Queue Management

#### GET `/queue/job/{job_id}`
Get the status of a specific job.

#### GET `/queue/jobs`
List all jobs with optional filtering.

**Query Parameters**:
- `limit`: Number of jobs to return (default: 50)
- `offset`: Pagination offset (default: 0)
- `status`: Filter by status (queued, processing, completed, failed, cancelled)

#### GET `/queue/stats`
Get comprehensive system statistics.

**Response**:
```json
{
  "total_jobs": 150,
  "active_jobs": 2,
  "queue_size": 5,
  "max_concurrent_jobs": 2,
  "status_counts": {
    "completed": 120,
    "queued": 5,
    "processing": 2,
    "failed": 3
  },
  "system_status": "healthy",
  "average_processing_time": 2.3,
  "webhook_success_rate": 0.98
}
```

#### DELETE `/queue/job/{job_id}`
Cancel a job (only queued or processing jobs).

#### POST `/queue/cleanup`
Clean up old completed/failed jobs.

**Query Parameters**:
- `max_age_hours`: Maximum age in hours (default: 24)

### Legacy Endpoints (Backward Compatibility)

#### POST `/describe`
Synchronous image description generation.

#### POST `/describe/file`
File upload for image description.

#### POST `/caption`
Generate image captions.

#### GET `/health`
System health check.

#### GET `/model/info`
Model information and status.

## Configuration

### Queue Settings
The queue manager can be configured in `app/services/queue_manager.py`:

```python
# Default settings
max_workers = 2          # Concurrent processing jobs
max_queue_size = 100     # Maximum queue size
```

### Priority Mapping
Motion intensity to priority mapping:
- `≥ 0.8`: Critical priority
- `≥ 0.6`: High priority  
- `≥ 0.4`: Normal priority
- `< 0.4`: Low priority

## Webhook Integration

### Webhook Payload Format
```json
{
  "job_id": "uuid-string",
  "camera_id": "camera_001",
  "status": "completed",
  "result": {
    "caption": "A person is walking through the front door",
    "confidence": 0.85,
    "processing_time": 2.3,
    "model_info": {
      "model": "Moondream2",
      "device": "cuda",
      "camera_id": "camera_001",
      "motion_intensity": 0.7,
      "priority": "high"
    }
  },
  "error_message": null,
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {
    "zone": "entrance",
    "sensitivity": "high"
  }
}
```

### Error Handling
If processing fails, the webhook will include an error message:
```json
{
  "job_id": "uuid-string",
  "camera_id": "camera_001",
  "status": "failed",
  "result": null,
  "error_message": "Model processing failed: CUDA out of memory",
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {}
}
```

## Usage Examples

### Python Client Example
```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
image = Image.open("motion_frame.jpg")
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
frame_data = base64.b64encode(buffer.getvalue()).decode()

# Send motion detection request
response = requests.post("http://localhost:8000/motion/detect", json={
    "camera_id": "camera_001",
    "frame_data": frame_data,
    "motion_intensity": 0.8,
    "location": "Front Door"
})

job_id = response.json()["job_id"]

# Check job status
status_response = requests.get(f"http://localhost:8000/queue/job/{job_id}")
print(status_response.json())
```

### Webhook Receiver Example
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    
    if data['status'] == 'completed':
        caption = data['result']['caption']
        camera_id = data['camera_id']
        print(f"Motion detected on {camera_id}: {caption}")
        
        # Send alert, store in database, etc.
        
    elif data['status'] == 'failed':
        print(f"Processing failed: {data['error_message']}")
    
    return {'status': 'received'}, 200

if __name__ == '__main__':
    app.run(port=5000)
```

## Performance Considerations

### Queue Management
- **Worker Count**: Adjust based on GPU memory and processing requirements
- **Queue Size**: Balance between memory usage and frame buffering
- **Priority Queuing**: Ensures critical events are processed first
- **Overflow Protection**: Drops low-priority jobs when queue is full

### Monitoring
- **System Stats**: Monitor queue depth, processing times, and success rates
- **Webhook Success Rate**: Track delivery reliability
- **Memory Usage**: Monitor GPU and system memory consumption
- **Processing Latency**: Track end-to-end processing times

## Troubleshooting

### Common Issues

1. **Queue Full**: Increase `max_queue_size` or implement frame sampling
2. **Slow Processing**: Increase `max_workers` or optimize model loading
3. **Webhook Failures**: Check network connectivity and webhook endpoint
4. **Memory Issues**: Reduce `max_workers` or implement memory cleanup

### Logs
Check `logs/moondream2.log` for detailed error messages and system information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the documentation
- Review the logs
- Open an issue on GitHub
