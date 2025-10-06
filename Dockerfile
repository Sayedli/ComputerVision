FROM python:3.10-slim

# Avoid Python writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for building dlib/face-recognition and for OpenCV runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libgl1 \
    libx11-6 libxcb1 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal metadata and dependencies first (better layer caching)
COPY requirements.txt requirements.txt
COPY requirements-ui.txt requirements-ui.txt
COPY requirements-django.txt requirements-django.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ui.txt \
    && pip install --no-cache-dir -r requirements-django.txt \
    && pip install --no-cache-dir -e .

# Copy source code (after deps for better caching)
COPY fr fr
COPY ui ui
COPY scripts scripts
COPY manage.py manage.py
COPY web web
COPY recognition recognition

# Ensure common mount points exist
RUN mkdir -p dataset models

# Default command shows CLI help. Override for specific actions.
CMD ["fr", "-h"]
