#!/bin/bash

if [ ! -f /etc/apt/sources.list.d/latentai-stable.list ]; then
    echo "Error: Add latentai apt repo before running this script."
    echo ""
    echo "Example:"
    echo " wget -qO - https://public.latentai.io/add_apt_repository | sudo bash"
    echo " sudo apt update"
    exit
fi

apt update

sudo apt install pylre

apt install -y libopencv-dev
apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
apt install -y liblre-cuda liblre-dev

OWD=$PWD
mkdir ~/.torch-apps
cd ~/.torch-apps
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
cd $OWD

apt install -y libjpeg-dev zlib1g-dev libavcodec-dev libpng-dev
git clone --branch v0.13.1 https://github.com/pytorch/vision torchvision
cd torchvision
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=~/.torch-apps/libtorch -DWITH_CUDA=ON .. 
make -j12
make install
ldconfig
