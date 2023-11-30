#!/bin/bash

remote_dir=hmblair@172.172.82.67:/home/hmblair # remote directory
folder=${PWD##*/} # get current folder name
parentdir="$(dirname "$PWD")" # get parent directory name
rsync -azh --progress --exclude 'Data/CSV' --exclude '*.ckpt' "$PWD" "$remote_dir" # copy current directory to remote directory
rsync -azh --progress --exclude 'Data/CSV' --exclude '*.ckpt' "$remote_dir"/"$folder" "$parentdir" # copy remote directory to current directory