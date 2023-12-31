# downloader.py

from tqdm import tqdm
import requests
import os 

import shutil
import warnings


if not os.path.exists('Data'):
    os.mkdir('Data')

def downloader_with_progress(url, filename):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f'Downloading {filename}')
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


import zipfile
import tarfile 
def unarchive(input_file : str, output_file : str):
    filetype = input_file.split('.')[-1]
    if input_file.endswith("tar"):
        with tarfile.open(input_file, "r:") as tar:
            tar.extractall(output_file)
    elif input_file.endswith("tar.gz") or input_file.endswith("tgz"):
        with tarfile.open(input_file, "r:gz") as tar:
            tar.extractall(output_file)
    elif input_file.endswith("tar.bz2") or input_file.endswith("tbz"):
        with tarfile.open(input_file, "r:bz2") as tar:
            tar.extractall(output_file)
    elif input_file.endswith("tar.xz") or input_file.endswith("txz"):
        with tarfile.open(input_file, "r:xz") as tar:
            tar.extractall(output_file)
    elif input_file.endswith("zip"):
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            zip_ref.extractall(output_file)
    else:
        raise OSError(f'Could not extract {input_file} as no appropriate extractor for the filetype "{filetype}" is found.')


def download_imagenet_tiny():
    # download the dataset
    zipped_file = 'Data/tiny-imagenet-200.zip'
    if not os.path.exists(zipped_file):
        url = 'https://image-net.org/data/tiny-imagenet-200.zip'
        downloader_with_progress(url, zipped_file)
    
    # extract the dataset
    if not os.path.exists('Data/tiny-imagenet-200'):
        print(f'Extracting {zipped_file}')
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall('Data/')
    
    # extract only the train dataset
    if not os.path.exists('Data/imagenet-tiny'):
        shutil.move('Data/tiny-imagenet-200/train', 'Data/imagenet-tiny')


def download_decathalon():
    # download the dataset
    zipped_file = 'Data/decathlon-1.0-data-imagenet.tar'
    if not os.path.exists(zipped_file):
        url = 'https://image-net.org/data/decathlon-1.0-data-imagenet.tar'
        downloader_with_progress(url, zipped_file)

    # extract the dataset
    if not os.path.exists('Data/decathlon'):
        print(f'Extracting {zipped_file}')
        with tarfile.open(zipped_file, 'r') as tar_ref:
            tar_ref.extractall('Data/')

    # rename the dataset
    if not os.path.exists('Data/decathlon'):
        shutil.move('Data/imagenet12', 'Data/decathlon')


def download_indoor_scenes():
    # download the dataset
    zipped_file = 'Data/indoorCVPR_09.tar'
    if not os.path.exists(zipped_file):
        url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        downloader_with_progress(url, zipped_file)

    # extract the dataset
    if not os.path.exists('Data/indoor_scenes'):
        print(f'Extracting {zipped_file}')
        with tarfile.open(zipped_file, 'r') as tar_ref:
            tar_ref.extractall('Data/')

    # rename the dataset
    if not os.path.exists('Data/indoor_scenes'):
        shutil.move('Data/Images', 'Data/indoor_scenes')


def download_fruits():
    # download the dataset
    zipped_file = 'Data/fruits-360_dataset.zip'
    if not os.path.exists(zipped_file):
        url = 'https://data.mendeley.com/public-files/datasets/rp73yg93n8/files/56487963-3773-495e-a4fc-c1862b6daf91/file_downloaded'
        downloader_with_progress(url, zipped_file)

    # extract the dataset
    if not os.path.exists('Data/fruits'):
        print(f'Extracting {zipped_file}')
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall('Data/')

    # rename the dataset
    if not os.path.exists('Data/fruits'):
        shutil.move('Data/fruits-360/Training', 'Data/fruits')


def downloader(dataset_name : str):
    if dataset_name == 'imagenet-tiny':
        download_imagenet_tiny()
        return os.path.join('Data', 'imagenet-tiny')
    elif dataset_name == 'decathlon':
        download_decathalon()
        return os.path.join('Data', 'decathlon')
    elif dataset_name == 'indoor_scenes':
        download_indoor_scenes()
        return os.path.join('Data', 'indoor_scenes')
    elif dataset_name == 'fruits':
        download_fruits()
        return os.path.join('Data', 'fruits')
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')
    








