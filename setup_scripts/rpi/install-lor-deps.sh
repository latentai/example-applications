#!/bin/bash

sudo apt update -y
sudo apt upgrade -y
pip3 install Cython
pip3 install setuptools==58.3.0
pip3 install gdown

# So that gdown can be found:
export PATH="$HOME/.local/bin:$PATH"

# To install PyTorch 1.13.0 for python & c++ support, use the whl below
sudo apt install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
gdown https://drive.google.com/uc?id=1uLkZzUdx3LiJC-Sy_ofTACfHgFprumSg
pip3 install torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
rm torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
# And to carefully match the torchvision install to the above torch version:
# Torchvision 0.14.0 python only
gdown https://drive.google.com/uc?id=1AhbkLqKd8EZO2pZV_g9aFZGHZo2Ubc3O
pip3 install torchvision-0.14.0a0+5ce4506-cp39-cp39-linux_aarch64.whl
rm torchvision-0.14.0a0+5ce4506-cp39-cp39-linux_aarch64.whl

# This is only needed for EfficientDet
pip3 install effdet

echo
echo Suggest you add the following to .bash_profile:
echo
echo export PATH="\$HOME/.local/bin:\$PATH"
