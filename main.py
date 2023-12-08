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
    parser.add_argument('--learning_rate_warmup_epochs', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--mlp_dim', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--way', type=int)
    parser.add_argument('--shot', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--subepoch_factor', type=int, default=2)
    parser.add_argument('--load_version', type=int)
    args = parser.parse_args()

    def get_model_name(args):
        """
        Returns a string summary of the model and its hyperparameters.
        """
        ignore_args = ['accelerator', 'batch_size', 'num_workers', 'mode', 'cache']
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

    log_dir = 'lightning_logs'

    #take model name from arguments, or if not specified create one.
    if not args.model_name:
        model_name = get_model_name(args)
    else:
        model_name = args.model_name

    model_folder = os.path.join(log_dir, model_name)

    #if not loading, setup for new model creation.
    if not args.load_version:
        checkpoint_path = None

    #else setup for model loading
    else:
        if args.load_version == 0:
            path_end = 'last.ckpt'
        else:
            path_end = 'last-v{}.ckpt'.format(args.load_version)

        checkpoint_path = os.path.join(model_folder, 'checkpoints', path_end)
        print("loading version {} from model {}".format(args.load_version, model_name))

    logger = TensorBoardLogger(log_dir, name=model_name, version=0)

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(model_folder, 'checkpoints'),
        filename='best',
        monitor='val_loss',
        mode='min',
        save_last=True,
        )

    trainer = pl.Trainer(
        accelerator = args.accelerator,
        max_epochs = args.max_epochs,
        precision = '16-mixed' if args.accelerator == 'gpu' else '32',
        logger = logger,
        callbacks = [model_checkpoint],
        )

    # get the data path
    if args.dataset == 'imagenet-tiny':
        from downloader import download_imagenet_tiny
        download_imagenet_tiny()
        train_path = os.path.join('Data', 'imagenet-tiny')
    elif args.dataset == 'omniglot':
        warnings.warn('There is no downloader for omniglot.')
        data_path = os.path.join('Data', 'omniglot_resized')
    elif args.dataset == 'decathalon':
        from downloader import download_decathalon
        download_decathalon()
        paths = {'train':  os.path.join('Data', 'decathlon', 'train'),
                 'val': os.path.join('Data', 'decathlon', 'val')}
    else:
        raise ValueError(f'Invalid dataset name {args.dataset}')

    # initialise the data module
    datamodule = ClipDataModule(
        paths = paths,
        batch_size = args.batch_size,
        way = args.way,
        shot = args.shot,
        num_workers = args.num_workers,
        cache = args.cache,
        subepoch_factor=args.subepoch_factor
        )

    # for the learning rate scheduler
    datamodule.setup('fit')
    steps_per_epoch = len(datamodule.train_dataloader())
    warmup_steps = args.learning_rate_warmup_epochs * steps_per_epoch

    # initialise the model
    if args.model == 'ProtoNetICL':
        model = ProtoNetICL(
            lr=args.learning_rate,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=datamodule.embedding_dim,
            mlp_dim=args.mlp_dim,
            warmup_steps=warmup_steps,
            )
    elif args.model == 'ProtoNetSkip':
        model = ProtoNetSkip(
            lr=args.learning_rate,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=datamodule.embedding_dim,
            mlp_dim=args.mlp_dim,
            warmup_steps=warmup_steps,
            )
    elif args.model == 'ProtoNetWithoutEncoder':
        model = ProtoNetWithoutEncoder(
            lr=args.learning_rate,
            )
    else:
        raise ValueError(f'Invalid model name {args.model}')

    # run the given mode
    if args.mode == 'train':
        trainer.fit(model, datamodule, ckpt_path = checkpoint_path)
    elif args.mode == 'test':
        trainer.test(model, datamodule, ckpt_path = checkpoint_path)
    else:
        raise ValueError(f'Invalid mode {args.mode}')
