#!/usr/bin/env bash

# CONFIG=$1
CONFIG="/data/home/jeongyeon/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.py"
GPUS=$2
# GPUS="1,3"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    # --nproc_per_node=2 \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
