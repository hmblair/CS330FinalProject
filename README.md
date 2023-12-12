Class-Scalable In-Context Meta-Learning.

Setup:

This project uses Python 3.11. To install the requirements with pip, run

pip install -r requirements.txt

There are two main programs, main.py and test.py, which perform
model training and testing, respectively.

To train our main model arhitecture ProtoNetICL on the ImageNet decathalaon dataset, run

python3 main.py \
    --model ProtoNetICL \
    --max_epochs 100 \
    --learning_rate 0.001 \
    --learning_rate_warmup 500 \
    --num_layers 2 \
    --num_heads 8 \
    --mlp_dim 128 \
    --batch_size 16 \
    --way 2 \
    --shot 2 \
    --dataset decathalon \
    --num_workers 0 \

To test a model on a the fruits dataset, create a folder [MODEL_FOLDER] in the main
directory, and place the log associated to the model in [MODEL_FOLDER]. You
can find this log in ./lightning_logs.

python3 test.py \
    --model_folder [MODEL_FOLDER] \
    --batch_size 16 \
    --way 2 5 10\
    --shot 1 5 10 \
    --dataset fruits \
    --num_workers 1

You can also test your model on the dataset indoor_scenes.
