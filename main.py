import pytorch_lightning as pl
import argparse

from protonet import ProtoNetICL, ProtoNetSkip, ProtoNetWithoutEncoder
from Data import ClipDataModule
import os
import warnings
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Scalable In-Context Meta-Learning')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--learning_rate_warmup', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--mlp_dim', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--way', type=int)
    parser.add_argument('--shot', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    def get_model_name(args):
        """
        Returns a string summary of the model and its hyperparameters.
        """
        ignore_args = ['accelerator', 'batch_size', 'num_workers', 'mode']
        return '_'.join(
            [str(getattr(args, arg)) 
                for arg in vars(args) 
                if getattr(args, arg) is not None
                and arg not in ignore_args
            ]
                )


    ## Priority:
    ## TODO: pass through random clip vectors to see hw it performs (Bhargav)
    ## TODO: Try using diffrent CLIP models (Hamish)
    ## TODO: Try a different LR scheduler (it must decrease significantly by 30 epochs or so) (Hamish)
    ## TODO: See what happens with CLIP only, and no encoder (Henry) (Change trainer.fit to trainer.test)
    ## TODO: Start making the poster in Google Slides (Henry)

    ## Lower priority
    ## TODO: Make the dataloader deterministic (so it loops over the entire dataset exactly once per epoch) (Hamish)
    ## TODO: Make the dataset work with multiple GPUS (Hamish)


    # get the model name from the hyperparameters
    model_name = get_model_name(args)

    # initialise the logger and checkpoint callback
    log_dir = 'lightning_logs'
    logger = TensorBoardLogger(log_dir, name=model_name, version=0)
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(log_dir, model_name, 'checkpoints'),
        filename='best',
        monitor='val_loss',
        mode='min',
        save_last=True
        )

    # initialise the trainer
    trainer = pl.Trainer(
        accelerator = args.accelerator,
        max_epochs = args.max_epochs,
        precision = '16-mixed' if args.accelerator == 'gpu' else '32',
        logger = logger,
        )

    # get the data path
    if args.dataset == 'imagenet-tiny':
        from downloader import download_imagenet_tiny
        download_imagenet_tiny()
        data_path = os.path.join('Data', 'imagenet-tiny')
    elif args.dataset == 'omniglot':
        warnings.warn('There is no downloader for omniglot.')
        data_path = os.path.join('Data', 'omniglot_resized')
    elif args.dataset == 'decathlon':
        from downloader import download_decathalon
        download_decathalon()
        data_path = os.path.join('Data', 'decathlon-1.0-data-imagenet')
    else: 
        raise ValueError(f'Invalid dataset name {args.dataset}')

    # initialise the data module
    datamodule = ClipDataModule(
        path = data_path,
        split = [0.8, 0.1, 0.1],
        batch_size = args.batch_size,
        way = args.way,
        shot = args.shot,
        num_workers = args.num_workers,
        )

    # initialise the model
    if args.model == 'ProtoNetICL':
        model = ProtoNetICL(
            lr=args.learning_rate,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=datamodule.embedding_dim,
            mlp_dim=args.mlp_dim,
            warmup_steps=args.learning_rate_warmup,
            )
    elif args.model == 'ProtoNetSkip':
        model = ProtoNetSkip(
            lr=args.learning_rate,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=datamodule.embedding_dim,
            mlp_dim=args.mlp_dim,
            warmup_steps=args.learning_rate_warmup,
            )
    elif args.model == 'ProtoNetWithoutEncoder':
        model = ProtoNetWithoutEncoder(
            lr=args.learning_rate,
            )
    else:
        raise ValueError(f'Invalid model name {args.model}')

    # run the given mode
    if args.mode == 'train':
        trainer.fit(model, datamodule)
    elif args.mode == 'test':
        trainer.test(model, datamodule)
    else:
        raise ValueError(f'Invalid mode {args.mode}')
    

