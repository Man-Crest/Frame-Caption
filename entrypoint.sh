#!/bin/bash

set -e

echo "🚀 Starting Moondream2 VLM Container..."

# Function to check if transformers is installed
check_transformers() {
    echo "🔍 Checking Transformers installation..."
    
    if python -c "import transformers; print('✅ Transformers is installed')" 2>/dev/null; then
        echo "✅ Transformers module found"
        return 0
    else
        echo "❌ Transformers module not found"
        echo "📦 Attempting to install Transformers..."
        
        if pip install --no-cache-dir transformers torch 2>/dev/null; then
            echo "✅ Transformers installation successful"
            if python -c "import transformers; print('✅ Transformers verified')" 2>/dev/null; then
                echo "✅ Transformers installation verified"
                return 0
            fi
        else
            echo "❌ Transformers installation failed"
        fi
        
        echo "❌ All installation methods failed"
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

# If the bundle contains a .onnx inside, it should be handled by the app loader expecting a .onnx path
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
    
    # Start the FastAPI server
    exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --log-level info \
        --access-log
}

# Main execution
main() {
    echo "🔧 Initializing Moondream2 VLM..."
    
    # Check GPU
    check_gpu
    
    # Check and install transformers if needed
    if ! check_transformers; then
        echo "❌ Failed to install Transformers. Starting server anyway..."
        echo "💡 The model will be downloaded on first request"
        echo "💡 You can manually install transformers later if needed"
    fi
    
    # Download model if needed
    download_model
    
    # Start application
    start_app
}

# Run main function
main
