#!/bin/bash
GPUS="device=1" # all
SHM=30g
IMGS=/home/z/work24/gits/Diffusion/mysamples/originals/1280_704
CHK="/mnt/Data/data/weights/NvidiaCosmos/checkpoints"
docker run --gpus $GPUS --cpuset-cpus=0-20 --network=host  -it --rm --shm-size $SHM \
    -v `pwd`:/app -v ${IMGS}:/app/image_1280_704 -v ${CHK}:/app/checkpoints --workdir /app \
    -e TORCH_EXTENSIONS_DIR=/app/tmp nvidia/xvdp/cosmos-predict1:latest