#!/bin/bash

# Some checks on readiness:

if which nvcc > /dev/null; then
    echo "Found nvcc in path"
else
    echo "Error: nvcc not found in path: $PATH"
    echo ""
    echo "Add nvcc to path before running this script"
    exit
fi

if [ ! -f /etc/apt/sources.list.d/latentai-stable.list ]; then
    echo "Error: Add latentai apt repo before running this script."
    echo ""
    echo "Example:"
    echo " wget -qO - https://public.latentai.io/add_apt_repository | sudo bash"
    echo " sudo apt update"
    exit
fi

sudo apt update

sudo apt install -y libopencv-dev
sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo apt install -y latentai-runtime-cuda latentai-runtime-dev

pip3 install --upgrade pip
pip3 install testresources
pip3 install scikit-build
pip3 install opencv-python==4.5.4.60
pip3 install gdown

