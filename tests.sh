#!/bin/bash

python3 test.py \
    --model_folder lightning_logs \
    --batch_size 16 \
    --way 2 5 10 25 \
    --shot 1 2 5 10 25 \
    --dataset indoor_scenes \
    --num_workers 1 