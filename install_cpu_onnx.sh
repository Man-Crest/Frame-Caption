#!/bin/bash

set -e

echo "🚀 Installing Moondream2-0.5B ONNX VLM for CPU-only systems"
echo "=========================================================="

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is not compatible. Please install Python 3.8 or higher."
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install CPU-optimized PyTorch
echo "📦 Installing CPU-optimized PyTorch..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ONNX Runtime for better CPU performance
echo "📦 Installing ONNX Runtime..."
python3 -m pip install onnxruntime

# Install other requirements
echo "📦 Installing other dependencies..."
python3 -m pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data

# Set permissions
chmod +x entrypoint.sh

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "🌙 Moondream2-0.5B ONNX VLM is ready for CPU-only usage"
echo "========================================================"
echo ""
echo "Key Features:"
echo "  - 0.5B parameter model (much smaller than 2B)"
echo "  - ONNX optimized for CPU performance"
echo "  - Reduced memory usage"
echo "  - Faster inference on CPU-only systems"
echo ""
echo "To start the server:"
echo "  python3 -m app.main"
echo ""
echo "Or using Docker:"
echo "  docker-compose up"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""
echo "Note: This installation uses the ONNX version of Moondream2-0.5B"
echo "which is optimized for CPU-only systems and provides better performance."
echo ""
echo "Model source: https://huggingface.co/vikhyatk/moondream2/blob/onnx/moondream-0_5b-int8.mf.gz"
