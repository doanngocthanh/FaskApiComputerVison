FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies - full python image has most libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements file
COPY requitements.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p .local/share/models/onnx && \
    mkdir -p .local/share/models/pth && \
    mkdir -p .local/share/models/pt && \
    mkdir -p .local/share/database && \
    mkdir -p temp

# Copy application code (excluding files in .dockerignore)
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Expose port
EXPOSE 8000

# Health check (run as root for curl access)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Switch to non-root user
USER app

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]