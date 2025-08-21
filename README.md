# Moondream2 VLM API with Queue Management

A high-performance Vision Language Model API server using Moondream2 0.5B model with simple queue management for image caption generation.

## 🚀 Features

- **Moondream2 0.5B Model**: CPU-optimized vision language model
- **Queue Management**: Asynchronous job processing with webhook support
- **Multiple Endpoints**: Caption generation, image description, and surveillance analysis
- **Docker Support**: Easy deployment with Docker Compose
- **CPU-Only**: No GPU required, runs efficiently on CPU

## 🔧 Configuration

The system is configured to use the Moondream2 0.5B model with the correct API pattern:

```python
import moondream as md
from PIL import Image

# Load the model
model = md.vl(model="path/to/moondream-0_5b-int8.mf")

# Load and encode image
image = Image.open("path/to/image.jpg")
encoded_image = model.encode_image(image)  # Only needs to be run once per image

# Generate caption
caption = model.caption(encoded_image)["caption"]
print("Caption:", caption)

# Ask a question (visual question answering)
answer = model.query(encoded_image, "What is in this image?")["answer"]
print("Answer:", answer)
```

## 🐳 Quick Start with Docker

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **The API will be available at:**
   - Main API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📋 API Endpoints

### Queue-based Caption Generation
- `POST /caption/generate` - Add caption job to queue
- `GET /caption/job/{job_id}` - Check job status
- `GET /caption/stats` - Get queue statistics

### Direct Image Processing
- `POST /describe` - Generate image description (synchronous)
- `POST /describe/file` - Process uploaded image file
- `POST /caption` - Generate image caption

### System Information
- `GET /health` - Health check
- `GET /model/info` - Model information

## 🔄 Queue System

The queue system provides asynchronous processing for high-throughput scenarios:

1. **Submit Job**: Send image data to `/caption/generate`
2. **Get Job ID**: Receive a job ID for tracking
3. **Check Status**: Poll `/caption/job/{job_id}` for status updates
4. **Webhook Delivery**: Optional webhook notifications when jobs complete

### Example Queue Usage

```bash
# Submit a caption job
curl -X POST "http://localhost:8000/caption/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_data",
    "prompt": "Describe what you see in this image",
    "webhook_url": "https://your-webhook-url.com/callback"
  }'

# Check job status
curl "http://localhost:8000/caption/job/{job_id}"

# Get queue statistics
curl "http://localhost:8000/caption/stats"
```

## 🛠️ Development

### Local Development Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the API:**
   ```bash
   python test_moondream_api.py
   ```

3. **Run the server:**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Environment Variables

- `MOONDREAM_BACKEND=mf` - Use Moondream .mf backend
- `MOONDREAM_MF_PATH=/app/models/moondream2-onnx/moondream-0_5b-int8.mf` - Model file path
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `TOKENIZERS_PARALLELISM=false` - Disable tokenizer parallelism

## 📊 Performance

- **Model Size**: 0.5B parameters (CPU-optimized)
- **Memory Usage**: ~2-4GB RAM
- **Processing Speed**: ~1-3 seconds per image (CPU)
- **Queue Capacity**: Configurable (default: 50 jobs)
- **Concurrent Jobs**: Configurable (default: 2 workers)

## 🔍 Troubleshooting

### Model Validation

The system includes comprehensive model validation to ensure the Moondream2 model file is properly downloaded and accessible:

#### Check Model Status
```bash
# Check model validation status
curl http://localhost:8000/model/validate

# Get detailed model information
curl http://localhost:8000/model/info
```

#### Common Model Issues

1. **Model File Missing**: The model file is not downloaded or in the wrong location
   - **Solution**: The container will automatically download the model on first run
   - **Check**: Verify `/app/models/moondream2-onnx/moondream-0_5b-int8.mf` exists

2. **Model File Empty**: The downloaded file is corrupted or incomplete
   - **Solution**: Remove the file and restart the container to re-download
   - **Command**: `docker-compose down && docker-compose up --build`

3. **Permission Issues**: The model file is not readable
   - **Solution**: Check file permissions inside the container
   - **Command**: `docker exec moondream2-vlm ls -la /app/models/moondream2-onnx/`

#### Manual Model Validation
```bash
# Quick status check (recommended)
python check_model_status.py

# Test API functionality
python test_moondream_api.py

# Check model file directly
docker exec moondream2-vlm ls -la /app/models/moondream2-onnx/
```

### Other Common Issues

1. **Memory Issues**: Reduce concurrent jobs or increase system RAM
2. **Queue Full**: Increase queue size or wait for jobs to complete
3. **Import Errors**: Ensure all dependencies are installed correctly

### Logs

Check container logs for detailed information:
```bash
docker-compose logs -f moondream2-vlm
```

## 📝 License

This project uses the Moondream2 model which is subject to its own license terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
