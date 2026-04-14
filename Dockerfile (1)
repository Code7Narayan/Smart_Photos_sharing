# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Modify requirements for server optimization
# Replace opencv-python with headless version for smaller size
# Modify requirements for server optimization
# (Removed sed command as requirements.txt already has opencv-python-headless)
# RUN sed -i 's/opencv-python/opencv-python-headless/g' requirements.txt

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace model during build to avoid runtime download
# Store in /app/.insightface instead of /root to allow non-root users (HuggingFace) to access
RUN mkdir -p /app/.insightface/models
RUN python -c "import insightface; \
    from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_l', root='/app/.insightface')"
    
# Fix permissions for Hugging Face (User 1000)
RUN chmod -R 777 /app/.insightface

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded models (Crucial step!)
COPY --from=builder /app/.insightface /app/.insightface

# Install runtime libs for OpenCV (headless needs minimal libs)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose the port (helper for some platforms)
EXPOSE 8000

# Run using the shell script
CMD ["./start.sh"]
