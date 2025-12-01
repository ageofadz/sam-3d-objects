# Base image – example
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# -------------------------
# 1. System deps (git first!)
# -------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip so MoGe + spconv don’t choke
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# 2. Python deps
# -------------------------
# Must match PyTorch version and CUDA version
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# -------------------------
# 3. Special packages
# -------------------------

# spconv matching CUDA 12.x (cu120)
RUN pip install --no-cache-dir "spconv-cu120==2.3.6"

# pytorch3d
RUN pip install --no-cache-dir "pytorch3d==0.7.8"

# MoGe (NOT on PyPI — must clone from GitHub)
RUN pip install --no-cache-dir \
    git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b