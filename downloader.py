# downloader.py

from tqdm import tqdm
import requests
import os 
import zipfile
import shutil

from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(parent_dir, 'Data')


def downloader_with_progress(url, filename):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")



def download_imagenet_tiny():
    # download the dataset
    zipped_file = 'tiny-imagenet-200.zip'
    if not os.path.exists(zipped_file):
        url = 'https://image-net.org/data/tiny-imagenet-200.zip'
        downloader_with_progress(url, zipped_file)
    
    # extract the dataset
    if not os.path.exists('tiny-imagenet-200'):
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall()
    
    # extract only the train dataset
    if not os.path.exists('imagenet-tiny'):
        shutil.move('tiny-imagenet-200/train', 'imagenet-tiny')







