FROM python:3.12.2-slim

# Install system dependencies required for DeepFace, OpenCV, and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for reference images
RUN mkdir -p reference_images

# Expose port
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV TF_USE_LEGACY_KERAS=1

# Run the application (using shell form to support environment variable)
CMD ["sh", "-c", "uvicorn index:app --host 0.0.0.0 --port ${PORT:-7860}"]