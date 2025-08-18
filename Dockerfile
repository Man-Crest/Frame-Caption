# Use Python base image for simplicity
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install and configure Git LFS
RUN git lfs install

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download moondream2 0.5B ONNX model from Hugging Face (optional, for faster startup)
RUN echo "📦 Pre-downloading Moondream2 0.5B ONNX model from Hugging Face..." && \
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='onnx', trust_remote_code=True, device_map='auto'); \
    tokenizer = AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='onnx', trust_remote_code=True); \
    print('✅ Moondream2 0.5B ONNX model pre-downloaded successfully')"

# Verify transformers installation
RUN python -c "import transformers; print('✅ Transformers verified during build')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/logs

# Set permissions
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
