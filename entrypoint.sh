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
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "📦 Downloading Moondream2 0.5B ONNX model file (mf.gz) from HuggingFace..."
        
        # Download the ONNX 0.5B archive using huggingface_hub without invoking transformers
        python - <<'PY'
import os
from huggingface_hub import hf_hub_download
import gzip
import shutil

os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'

repo_id = 'vikhyatk/moondream2'
revision = 'onnx'
filename = 'moondream-0_5b-int8.mf.gz'

os.makedirs('/app/models/moondream2-onnx', exist_ok=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir='/app/models/moondream2-onnx')
print(f'✅ Downloaded {filename} to {path}')

# Decompress to .mf (container-friendly)
mf_path = path[:-3] if path.endswith('.gz') else path
if path.endswith('.gz'):
    with gzip.open(path, 'rb') as f_in, open(mf_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f'✅ Decompressed to {mf_path}')

# Try to locate an embedded .onnx file if present (best-effort)
onnx_guess = os.path.join('/app/models/moondream2-onnx', 'moondream-0_5b-int8.onnx')
if not os.path.exists(onnx_guess):
    # Leave a marker file to indicate the expected path and that extraction is needed
    with open(os.path.join('/app/models/moondream2-onnx', 'README.txt'), 'w') as f:
        f.write('Place the extracted .onnx file here as moondream-0_5b-int8.onnx or update MOONDREAM_ONNX_PATH.')
PY
    else
        echo "✅ Model already exists at $MODEL_PATH"
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

# Function to start the application
start_app() {
    echo "🌙 Starting Moondream2 VLM API server..."
    
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
