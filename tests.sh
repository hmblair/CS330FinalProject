#!/bin/bash

python3 test.py \
    --model_folder logs_for_test_dec4 \
    --batch_size 16 \
    --way 2 5 10 25 \
    --shot 1 2 5 10 25 \
    --dataset fruits \
    --num_workers 1
