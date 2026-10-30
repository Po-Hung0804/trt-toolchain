FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# OpenCV needs these shared libraries even when no display is used
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/proj

COPY requirements.txt .
RUN pip install -r requirements.txt
