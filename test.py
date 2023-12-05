import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import argparse

from protonet import ProtoNetICL, ProtoNetSkip, ProtoNetWithoutEncoder
from Data import ClipDataModule
import os
import warnings
from itertools import product

if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')

    parser = argparse.ArgumentParser(prog='Scalable In-Context Meta-Learning')
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--way', type=int, nargs='+')
    parser.add_argument('--shot', type=int, nargs='+')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # get the data path
    if args.dataset == 'imagenet-tiny':
        from downloader import download_imagenet_tiny
        download_imagenet_tiny()
        data_path = os.path.join('Data', 'imagenet-tiny')
        paths = {'test' : data_path}
    elif args.dataset == 'omniglot':
        warnings.warn('There is no downloader for omniglot.')
        data_path = os.path.join('Data', 'omniglot_resized')
    elif args.dataset == 'decathlon':
        from downloader import download_decathalon
        download_decathalon()
        data_path = os.path.join('Data', 'decathlon')
        paths = {'test' : data_path}
    elif args.dataset == 'indoor_scenes':
        from downloader import download_indoor_scenes
        download_indoor_scenes()
        data_path = os.path.join('Data', 'indoor_scenes')
        paths = {'test' : data_path}
    else: 
        raise ValueError(f'Invalid dataset name {args.dataset}')
    
    # initialise the trainer
    trainer = pl.Trainer(
        accelerator = args.accelerator,
        precision = '16-mixed' if args.accelerator == 'gpu' else '32',
        logger = False,
        )

    for way, shot in product(args.way, args.shot):
        # initialise the data module
        datamodule = ClipDataModule(
            paths = paths,
            batch_size = args.batch_size,
            way = way,
            shot = shot,
            num_workers = args.num_workers,
            )
        

        import pandas as pd
        data = pd.DataFrame(columns=['model', 'way', 'shot', 'test_accuracy'])
        
        # test the models in the given folder
        for dir in os.listdir(args.model_folder):
            if os.path.isfile(dir):
                continue
            model_name = dir.split('_')[0]
            ckpt_dir = os.path.join(args.model_folder, dir, 'checkpoints', 'last.ckpt')

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
            print(f'Testing {dir} on {args.dataset} with {way}-way {shot}-shot')

            df2 = pd.DataFrame(
                data = [[dir, way, shot, trainer.test(model, datamodule)[0]['test_accuracy_epoch']]],
                columns = ['model', 'way', 'shot', 'test_accuracy']
            )
            data = pd.concat([data, df2], ignore_index=True)


        # test ProtoNetWithoutEncoder
        print(f'Testing ProtoNetWithoutEncoder on {args.dataset} with {args.way}-way {args.shot}-shot')
        model = ProtoNetWithoutEncoder()
        df2 = pd.DataFrame(
            data = [['ProtoNetWithoutEncoder', way, shot, trainer.test(model, datamodule)[0]['test_accuracy_epoch']]],
            columns = ['model', 'way', 'shot', 'test_accuracy']
        )
        data = pd.concat([data, df2], ignore_index=True)

    data.to_csv('results.csv')

