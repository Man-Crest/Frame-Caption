FROM python:3.11-slim

# System deps (minimal for CPU-only)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY entrypoint.sh . 

# Fix line endings and set permissions
RUN sed -i 's/\r$//' entrypoint.sh && \
    chmod +x entrypoint.sh

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Run entrypoint script directly
ENTRYPOINT ["./entrypoint.sh"]
