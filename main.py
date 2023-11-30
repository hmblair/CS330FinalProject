import pytorch_lightning as pl
import argparse

from Model import ProtoNetICL
from Data import ClipDataModule
import os
import warnings

parser = argparse.ArgumentParser(prog='Scalable In-Context Meta-Learning')
parser.add_argument('--max_epochs', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--num_layers', type=int)
parser.add_argument('--num_heads', type=int)
parser.add_argument('--mlp_dim', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--way', type=int)
parser.add_argument('--shot', type=int)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

# initialise the trainer
trainer = pl.Trainer(
    accelerator = 'cpu',
    max_epochs = args.max_epochs,
    precision = '32',
    )

# get the data path
if args.dataset == 'imagenet-tiny':
    from downloader import download_imagenet_tiny
    download_imagenet_tiny()
    data_path = os.path.join('Data', 'imagenet-tiny')
elif args.dataset == 'omniglot':
    warnings.warn('There is no downloader for omniglot.')
    data_path = os.path.join('Data', 'omniglot_resized')
else: 
    raise ValueError(f'Invalid dataset name {args.dataset}')

# initialise the data module
datamodule = ClipDataModule(
    path = data_path,
    split = [0.8, 0.1, 0.1],
    batch_size = args.batch_size,
    way = args.way,
    shot = args.shot,
    num_workers = 0
    )

# initialise the model
model = ProtoNetICL(
    lr=args.learning_rate,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    hidden_dim=datamodule.embedding_dim,
    mlp_dim=args.mlp_dim,
    )

# train the model
trainer.fit(model, datamodule)

