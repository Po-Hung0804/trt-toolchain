#!/bin/bash
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e YOLO_CONFIG_DIR=/workspace/proj/.ultralytics \
  -v ~/projects/trt-toolchain:/workspace/proj \
  trt-toolchain:dev "$@"
