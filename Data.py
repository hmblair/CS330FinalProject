import os
import random
from typing import Union, Optional, Tuple, Iterable

import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset
import clip
from PIL import Image

from base import BaseDataModule
from random import shuffle
from distributed_helper import distributed_breakpoint


class MetaLearningClipIterableDataset(IterableDataset):
    """
    IterableDataset for loading the Omniglot dataset (and hopefully more in the future) for meta-learning with CLIP.

    Args:
        data_folders (list[str]): The list of folders containing the data.
        way (int): The number of classes per batch.
        shot (int): The number of examples per class per batch.
        batch_size (int): The size of the batch.
        cache (bool): Whether to cache the images in memory.

    Methods:
        load_and_preprocess: Loads and preprocesses a single meta-train unit of data.
        _sample: Samples a single meta-train unit of data from the dataset. The last class is the query class.
        __iter__: Creates an iterator object that iterates over the data folders, yielding batches of data.
        __len__: Calculates the number of batches in the dataset.

    Attributes:
        way (int): The number of classes per batch.
        shot (int): The number of examples per class per batch.
        batch_size (int): The size of the batch.
        data_folders (list[str]): The list of folders containing the data.
        image_caching (bool): Whether to cache the images in memory.
        stored_images (dict[str, torch.Tensor]): Dictionary of image paths to cached images.
        preprocess (Callable): The CLIP preprocessing function.
    """
    def __init__(self,
                 data_folders : list[str],
                 way : int,
                 shot : int,
                 batch_size : int,
                 cache : bool = True):
        self.way = way # number of classes per batch
        self.shot = shot # number of examples per class
        self.batch_size = batch_size # size of the batch
        self.data_folders = data_folders # list of folders containing the data

        self.image_caching = cache # whether to cache images in memory
        self.stored_images = {}

        # load the clip preprocessing function
        encode, preprocess = clip.load("ViT-B/32") 
        self.preprocess = preprocess
        self.encode = encode
    


    def _get_worker_info(self) -> tuple[int, int]:
        """
        Gets the id of the current worker and the total number of workers.

        Returns:
            tuple[int, int]: The id of the current worker and the total number of workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = (0 if worker_info is None else worker_info.id)
        num_workers = (1 if worker_info is None else worker_info.num_workers)
        return worker_id, num_workers


    def load_and_preprocess(self, path : str) -> torch.Tensor:
        """
        Loads self.shot images from the specified path and preprocesses them
        using the CLIP preprocessing function.

        Args:
            path (str): The path to the folder containing the images.

        Returns:
            torch.Tensor: The preprocessed data.
        """
        # sample self.shot images from the class folder
        sampler = lambda x: random.sample(x, self.shot) 
        filenames = sampler(os.listdir(path))

        # load and preprocess the images
        files = []
        for filename in filenames:
            full_path = os.path.join(path, filename)
            if self.image_caching and full_path in self.stored_images:
                files.append(
                    self.stored_images[full_path]
                )
            else:
                file = self.preprocess(Image.open(full_path))
                if self.image_caching:
                    self.stored_images[full_path] = file
                files.append(file)

        return torch.stack(files)
    

    def _sample(self) -> torch.Tensor:
        """
        Samples a single meta-train unit of data from the dataset. The last class is the query class.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, way + 1, shot, feature_dim) containing the images.
        """
        files = []
        for _ in range(self.way):
            # sample a class folder
            class_folder = random.choice(self.data_folders)

            # sample self.shot images from the class folder
            files.append(self.load_and_preprocess(class_folder))

        # sample self.shot images from the final class folder, which is the query class
        files.append(self.load_and_preprocess(class_folder))

        # get indices for the classes, including the query class
        labels = list(range(self.way))
        labels += [labels[-1]]
        
        return torch.stack(files), torch.tensor(labels)


    def __iter__(self) -> Iterable: 
        """
        Creates an iterator object that iterates over the data folders, yielding batches of data.

        Returns:
            Iterable: The data iterator.
        """
        worker_id, num_workers = self._get_worker_info()
        self.data_folders = self.data_folders[worker_id::num_workers] # split the data folders across the workers
        while True:
            yield self._sample()


    def __len__(self) -> int:
        """
        Calculates the number of batches in the dataset.

        Returns:
            int: The number of batches in the dataset.
        """
        # get the number of workers; we need to scale the length by this since each worker will
        # increment the iterator by one
        _, num_workers = self._get_worker_info()

        # calculate the number of files in the dataset
        num_files = sum([len([name for name in os.listdir(path)]) for path in self.data_folders])
        return num_files // (self.way * self.shot * self.batch_size * num_workers)
    


class ClipDataModule(BaseDataModule):
    """
    DataModule for loading the Omniglot dataset (and hopefully more in the future) for meta-learning with CLIP.

    Args:
        path (Union[str, os.PathLike]): The path to the dataset directory.
        way (int): The number of classes per episode.
        shot (int): The number of examples per class per episode.
        *args: Additional positional arguments to be passed to the BaseDataModule constructor.
        **kwargs: Additional keyword arguments to be passed to the BaseDataModule constructor.

    Methods:
        _create_datasets: Creates the MetaLearningClipIterableDatasets for the train/val/test phases.
        _encode: Encodes the input tensor using CLIP.
        on_after_batch_transfer: Encodes the batch using CLIP after it has been transferred to the device.

    Attributes:
        paths (str | os.PathLike): The path to the directory that contains the image folders.
        way (int): The number of classes per episode.
        shot (int): The number of examples per class per episode.
        encode (Callable): The CLIP model.
        embedding_dim (int): The dimension of the CLIP embeddings.
    """
    def __init__(self,
                 paths : dict[str, str],
                 way : int,
                 shot : int,
                 cache : bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        for phase in paths.keys():
            if phase not in ['train', 'val', 'test']:
                raise ValueError(f'Invalid phase {phase}.')
        self.paths = paths
        self._verify_paths()

        # get the list of class folders for each phase
        self.class_folders = {phase : self._get_folders(phase) for phase in paths.keys()}
        
        # store the way and shot
        self.way = way
        self.shot = shot
        self.cache = cache

        # initialize the CLIP model and get the embedding dimension
        encode, _ = clip.load('ViT-B/32')
        self.encode = encode.encode_image
        self.embedding_dim = encode.visual.output_dim

        self.use_random_embeddings = False


    def _verify_paths(self):
        """
        Verify that the provided paths exist and are directories.

        Raises:
            OSError: If the path does not exist or is not a directory.
        """
        for phase, path in self.paths.items():
            if not os.path.exists(path):
                raise OSError(f'The specified path for phase {phase} does not exist.')
            if not os.path.isdir(path):
                raise OSError(f'The specified path for phase {phase} is not a directory.')


    def _get_folders(self, phase : str) -> list[str]:
        """
        Walk over the dataset directory and get the list of class folders, which 
        are the leaf subdirectories.

        Returns:
            list[str]: The list of class folders.
        """
        folders = []
        for dirpath, dirnames, filenames in os.walk(self.paths[phase]):
            if not dirnames:
                folders.append(dirpath)
        return folders


    def _create_datasets(self, phase: str):
        """
        Create MetaLearningClipIterableDatasets for the specified phase.

        Args:
            phase (str): The phase of the datasets.

        Returns:
            MetaLearningClipIterableDataset: The created dataset.
        """
        if phase not in self.paths.keys():
            return
        else:
            return MetaLearningClipIterableDataset(
                data_folders=self.class_folders[phase],
                way=self.way,
                shot=self.shot,
                batch_size=self.batch_size,
                cache=self.cache,
            )


    def _encode(self, x : torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor using CLIP.
        
        Args:
            x (torch.Tensor): The input tensor to be encoded, of size (*d, c, h, w)
        
        Returns:
            torch.Tensor: The encoded tensor, of size (*d, clip_embedding_dim)
        """
        *d, c, h, w = x.shape # get the shape of the input tensor
        x = x.reshape(-1, c, h, w) # reshape the input tensor to be 4D
        with torch.no_grad(): # disable gradient tracking
            x = self.encode(x) # encode using CLIP
        return x.reshape(*d, -1) # reshape the output tensor


    def on_after_batch_transfer(self, batch : torch.Tensor, dataloader_idx : int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the batch using CLIP after it has been transferred to the device.

        Args:
            batch (torch.Tensor): The batch of data.
            dataloader_idx (int): The index of the dataloader.

        Returns:
                tuple[torch.Tensor, torch.Tensor]: The encoded batch and the corresponding labels.
        """
        if self.use_random_embeddings:
            return self._random_clip_embeddings(), batch[1]
        else:
            return self._encode(batch[0]), batch[1]


    ## BHARGAV: Implement this
    def _random_clip_embeddings(self):
        """
        Generate random clip embeddings for the classes.
        """
        raise NotImplementedError




## A quick test to make sure that it works
def test():
    path = '/Users/hmblair/Documents/University/Graduate/Classes/CS330/FinalProject/Data/imagenet-tiny'
    datamodule = ClipDataModule(paths = {'train' : path},
                                batch_size = 4,
                                way = 2,
                                shot = 2,
                                num_workers = 0)

    datamodule.setup('fit')

    train_dataloader = datamodule.train_dataloader()

    for batch in train_dataloader:
        batch = datamodule.on_after_batch_transfer(batch, 0)
        breakpoint()


if __name__ == '__main__':
    test()