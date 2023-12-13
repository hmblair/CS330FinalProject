Class-Scalable In-Context Meta-Learning.

Setup:

This project uses Python 3.11. To install the requirements with pip, run

```
pip install -r requirements.txt
```

There are two main programs, main.py and test.py, which perform
model training and testing, respectively.

To train our main model arhitecture ProtoNetICL with 1 layer on the ImageNet decathalaon dataset on 2-way 2-shot support sets, run

```
python3 main.py \
    --model ProtoNetICL \
    --max_epochs 100 \
    --learning_rate 0.001 \
    --learning_rate_warmup 500 \
    --num_layers 1 \
    --num_heads 8 \
    --mlp_dim 128 \
    --batch_size 16 \
    --way 2 \
    --shot 2 \
    --dataset decathalon \
    --num_workers 0 \
    --subepoch_factor 16
```

To test a model on dataset, create a folder [MODEL_FOLDER] in the main
directory, and place the log associated to the model in [MODEL_FOLDER]. You
can find this log in ./lightning_logs. Here is an example where we test
our model on the fruits dataset with varying way (2,5,10) and shot (1,5,10).
All combinations will be tested.

```
python3 test.py \
    --model_folder [MODEL_FOLDER] \
    --batch_size 16 \
    --way 2 5 10 \
    --shot 1 5 10 \
    --dataset fruits \
    --num_workers 1
```

You can also test your model on the dataset indoor_scenes. Replace the dataset argument with 
```
--dataset indoor_scenes
```

Datasets are automatically downloaded. If you would like to download them without
initializing a training or test run, you can run individual functions
in the file downloader.py. For example, if you want to download the fruits
dataset, you can run the following python code from the main directory:

```
from downloader import download_fruits
download_fruits()
```
