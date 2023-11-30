#!/bin/bash

python3 main.py \
    --model ProtoNetICL \
    --max_epochs 100 \
    --learning_rate 0.001 \
    --num_layers 1 \
    --num_heads 8 \
    --mlp_dim 128 \
    --batch_size 16 \
    --way 5 \
    --shot 2 \
    --dataset imagenet-tiny