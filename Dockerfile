# Use official CUDA-enabled PyTorch runtime image (includes Python, CUDA, cuDNN, and torch)
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

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

# Upgrade pip and verify torch CUDA availability
RUN pip install --upgrade pip && \
    python -c "import torch; print('✅ Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download moondream2 model from Hugging Face (optional, for faster startup)
RUN echo "📦 Pre-downloading Moondream2 model from Hugging Face..." && \
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='2025-06-21', trust_remote_code=True, device_map='auto'); \
    tokenizer = AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='2025-06-21', trust_remote_code=True); \
    print('✅ Moondream2 model pre-downloaded successfully')"

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
