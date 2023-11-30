#!/bin/bash

python3 main.py \
    --max_epochs 100 \
    --learning_rate 0.001 \
    --num_layers 1 \
    --num_heads 8 \
    --mlp_dim 3 \
    --batch_size 2 \
    --way 5 \
    --shot 1\
    --dataset imagenet-tiny