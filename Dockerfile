FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.12 via Deadsnakes PPA
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Set python3 alias
RUN ln -nsf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -nsf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Install PyTorch 2.5.1 (Stable) compatible with CUDA 12.4
RUN pip install "torch==2.5.1" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy project files
COPY . /app

# Install dependencies
# 1. Install project in editable mode (handles dependencies in pyproject.toml if present, or just basic setup)
# 2. Install requirements.txt
# 3. Install Flash Attention 2 (compiled from source, takes time but critical for performance)
RUN pip install -e . && \
    pip install -r requirements.txt && \
    pip install flash-attn --no-build-isolation

# Default command
CMD ["/bin/bash"]
