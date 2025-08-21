#!/bin/bash

set -e

echo "🚀 Starting Moondream2 VLM Container..."

# Function to check if transformers is installed
check_transformers() {
    echo "🔍 Checking Moondream package installation..."
    if python -c "import moondream; print('✅ moondream is installed')" 2>/dev/null; then
        echo "✅ moondream module found"
        return 0
    else
        echo "❌ moondream module not found"
        echo "📦 Attempting to install moondream..."
        if pip install --no-cache-dir moondream 2>/dev/null; then
            echo "✅ moondream installation successful"
            return 0
        fi
        echo "❌ moondream installation failed"
        return 1
    fi
}

# Function to download model if not exists
download_model() {
    echo "📥 Checking for Moondream2 0.5B ONNX model..."
    
    MODEL_PATH="/app/models/moondream2-onnx"
    MODEL_FILE="/app/models/moondream2-onnx/moondream-0_5b-int8.mf"
    
    # Check if model file already exists and is valid
    if [ -f "$MODEL_FILE" ] && [ -s "$MODEL_FILE" ]; then
        echo "✅ Model file already exists and is valid: $MODEL_FILE"
        echo "📊 Model file size: $(du -h "$MODEL_FILE" | cut -f1)"
        return 0
    fi
    
    echo "📦 Downloading Moondream2 0.5B ONNX model file (mf.gz) from HuggingFace..."
    
    # Create model directory
    mkdir -p "$MODEL_PATH"
    
    # Download the ONNX 0.5B archive using huggingface_hub without invoking transformers
    python - <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download
import gzip
import shutil

os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'

repo_id = 'vikhyatk/moondream2'
revision = 'onnx'
filename = 'moondream-0_5b-int8.mf.gz'

try:
    print(f'📥 Downloading {filename} from {repo_id}...')
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir='/app/models/moondream2-onnx')
    print(f'✅ Downloaded {filename} to {path}')
    
    # Decompress to .mf (container-friendly)
    mf_path = path[:-3] if path.endswith('.gz') else path
    if path.endswith('.gz'):
        print(f'🔄 Decompressing {path} to {mf_path}...')
        with gzip.open(path, 'rb') as f_in, open(mf_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f'✅ Decompressed to {mf_path}')
        
        # Remove the compressed file to save space
        os.remove(path)
        print(f'🗑️  Removed compressed file: {path}')
    
    # Verify the final file
    if os.path.exists(mf_path) and os.path.getsize(mf_path) > 0:
        print(f'✅ Model file ready: {mf_path} ({os.path.getsize(mf_path)} bytes)')
    else:
        print(f'❌ Model file verification failed: {mf_path}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Download failed: {e}')
    sys.exit(1)
PY
    
    if [ $? -eq 0 ]; then
        echo "✅ Model download and setup completed successfully"
        echo "📊 Final model file size: $(du -h "$MODEL_FILE" | cut -f1)"
    else
        echo "❌ Model download failed"
        return 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        echo "⚠️  No GPU detected, running on CPU"
    fi
}

# Function to validate model before starting
validate_model_before_start() {
    echo "🔍 Validating model before starting application..."
    
    MODEL_FILE="/app/models/moondream2-onnx/moondream-0_5b-int8.mf"
    
    if [ -f "$MODEL_FILE" ] && [ -s "$MODEL_FILE" ]; then
        echo "✅ Model file exists and is valid: $MODEL_FILE"
        echo "📊 Model file size: $(du -h "$MODEL_FILE" | cut -f1)"
        return 0
    else
        echo "❌ Model file validation failed: $MODEL_FILE"
        return 1
    fi
}

# Function to start the application
start_app() {
    echo "🌙 Starting Moondream2 VLM API server..."
    
    # Validate model before starting
    if ! validate_model_before_start; then
        echo "❌ Cannot start application due to model validation failure"
        exit 1
    fi
    
    # Set environment variables for optimal performance
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export TOKENIZERS_PARALLELISM=false
    export PYTHONPATH=/app
    
    # Start the FastAPI server
    exec uvicorn app.main:app \
        --app-dir /app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --log-level info \
        --access-log
}

# Main execution
main() {
    echo "🔧 Initializing Moondream2 VLM..."
    
    # Check GPU (optional)
    check_gpu
    
    # Ensure moondream package is present
    if ! check_transformers; then
        echo "❌ Failed to install moondream. Server cannot start without it."
    fi
    
    # Download model if needed
    download_model
    
    # Start application
    start_app
}

# Run main function
main
