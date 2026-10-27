FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace/proj

COPY requirements.txt .
RUN pip install -r requirements.txt
