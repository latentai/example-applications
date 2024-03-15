#!/bin/bash

if [ ! -f /etc/apt/sources.list.d/latentai-stable.list ]; then
    echo "Error: Add latentai apt repo before running this script."
    echo ""
    echo "Example:"
    echo " wget -qO - https://public.latentai.io/add_apt_repository | sudo bash"
    echo " sudo apt update"
    exit
fi

sudo apt update -y
sudo apt upgrade -y
sudo apt install -y cmake
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

# pylre only required when using python or LOR
sudo apt install -y pylre

sudo apt install -y libopencv-dev
sudo apt install -y liblre-cpu liblre-dev

# Installed in the lor setup script, not required here
# sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

pip3 install opencv-python
pip3 install albumentations

echo
echo Suggest you add the following to .bash_profile:
echo
echo export PATH="\$HOME/.local/bin:\$PATH"
