#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/attention_pooling/models/acn/log \
    mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/attention_pooling/models/vgg_16_rgb_solver.prototxt \
    --weights=vgg_16_action_rgb_pretrain.caffemodel

