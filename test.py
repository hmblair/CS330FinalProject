import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import argparse

from protonet import ProtoNetICL, ProtoNetSkip, ProtoNetWithoutEncoder
from Data import ClipDataModule
import os
import warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Scalable In-Context Meta-Learning')
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--way', type=int)
    parser.add_argument('--shot', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--clip_only', action='store_true')
    args = parser.parse_args()


    # get the data path
    if args.dataset == 'imagenet-tiny':
        from downloader import download_imagenet_tiny
        download_imagenet_tiny()
        data_path = os.path.join('Data', 'imagenet-tiny')
        paths = {'test' : data_path}
    elif args.dataset == 'indoor_scenes':
        from downloader import download_indoor_scenes
        download_indoor_scenes()
        data_path = os.path.join('Data', 'indoor_scenes')
        paths = {'test' : data_path}
    elif args.dataset == 'fruits':
        from downloader import download_fruits
        download_fruits()
        data_path = os.path.join('Data', 'fruits')
        paths = {'test' : data_path}
    else: 
        raise ValueError(f'Invalid dataset name {args.dataset}')

    # initialise the data module
    datamodule = ClipDataModule(
        paths = paths,
        batch_size = args.batch_size,
        way = args.way,
        shot = args.shot,
        num_workers = args.num_workers,
        )
    
    # test the models in the given folder
    if not args.clip_only:
        for dir in os.listdir(args.model_folder):
            if os.path.isfile(dir):
                continue
            model_name = dir.split('_')[0]
            ckpt_dir = os.path.join(args.model_folder, dir, 'checkpoints', 'best.ckpt')

            # initialise the trainer
            log_dir = 'test_logs'
            logger = CSVLogger(log_dir, name=dir, version=0)
            trainer = pl.Trainer(
                accelerator = args.accelerator,
                precision = '16-mixed' if args.accelerator == 'gpu' else '32',
                logger = logger,
                )

            # initialise the model
            if model_name == 'ProtoNetICL':
                model = ProtoNetICL
            elif model_name == 'ProtoNetWithoutEncoder':
                model = ProtoNetWithoutEncoder
            else:
                raise ValueError(f'Invalid model name {model_name}')

            # load the model
            model = model.load_from_checkpoint(ckpt_dir)

            # test the model
            print(f'Testing {dir} on {args.dataset} with {args.way}-way {args.shot}-shot')
            trainer.test(model, datamodule)

    # test ProtoNetWithoutEncoder
    log_dir = 'test_logs'
    logger = CSVLogger(log_dir, name='ProtoNetWithoutEncoder', version=0)
    trainer = pl.Trainer(
        accelerator = args.accelerator,
        precision = '16-mixed' if args.accelerator == 'gpu' else '32',
        logger = logger,
        )
    
    print(f'Testing ProtoNetWithoutEncoder on {args.dataset} with {args.way}-way {args.shot}-shot')
    model = ProtoNetWithoutEncoder()
    trainer.test(model, datamodule)
