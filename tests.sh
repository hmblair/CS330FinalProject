#!/bin/bash

python3 test.py \
    --model_folder lightning_logs \
    --batch_size 16 \
    --way 5 \
    --shot 2 \
    --dataset imagenet-tiny \
    --num_workers 0 \
    --accelerator cpu \


python3 test.py \
    --model_folder lightning_logs \
    --batch_size 16 \
    --way 5 \
    --shot 5 \
    --dataset imagenet-tiny \
    --num_workers 0 \
    --accelerator cpu \