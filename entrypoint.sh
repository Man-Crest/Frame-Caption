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
        echo "📦 Downloading Moondream2 0.5B ONNX model from HuggingFace..."
        
        # Check if transformers is available
        if ! python -c "import transformers" 2>/dev/null; then
            echo "❌ Transformers not available, skipping model download"
            echo "💡 Model will be downloaded on first API request"
            return 0
        fi
        
        python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set model cache directory
os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'

# Download and cache the model from HuggingFace
print('Downloading Moondream2 0.5B ONNX model from HuggingFace...')
model = AutoModelForCausalLM.from_pretrained(
    'vikhyatk/moondream2',
    revision='onnx',
    trust_remote_code=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(
    'vikhyatk/moondream2', 
    revision='onnx',
    trust_remote_code=True
)
print('Model downloaded successfully!')
"
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
