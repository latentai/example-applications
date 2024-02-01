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

# OS dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip unzip
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

# Needed for LOR/Evaluate
sudo apt install -y libjpeg62 libjpeg8-dev

pip3 install --upgrade pip
pip3 install Cython
pip3 install -U --no-deps numpy==1.19.4 protobuf==3.19.6 pybind11 pkgconfig
pip3 install Pillow==8.4.0

# Installing cmake using snap
sudo apt remove -y --purge cmake
hash -r
sudo apt install -y snapd
sudo snap install cmake --classic
hash -r

# Alternative method of updating cmake if snap is not available (e.g. docker containers):
#
#wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
#sudo apt install -y software-properties-common
#sudo add-apt-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
#sudo apt install -y cmake

# Pytorch and Torchvision
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install -y libopenblas-base libopenmpi-dev
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libpng-dev
git clone --branch v0.11.3 https://github.com/pytorch/vision torchvision
(cd torchvision || exit && sudo python3 setup.py install)

# Install Torchvision for c++
mkdir torchvision/build_cpp
cd torchvision/build_cpp
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DWITH_CUDA=on ..
make -j4
sudo make install
sudo ldconfig
cd ../..

# Required to patch torch-1.10.0
if [ -f ~/.local/lib/python3.6/site-packages/torch/include/ATen/DLConvertor.h ]
then
    sed -i 's/TORCH_API DLContext/TORCH_API DLDevice/1' ~/.local/lib/python3.6/site-packages/torch/include/ATen/DLConvertor.h
fi

if [ -f /usr/local/lib/python3.6/dist-packages/torch/include/ATen/DLConvertor.h ]
then
    sudo sed -i 's/TORCH_API DLContext/TORCH_API DLDevice/1' /usr/local/lib/python3.6/dist-packages/torch/include/ATen/DLConvertor.h
fi

# Needed for >= LEIP 2.8
pip3 install typing-extensions==3.10.0.2
pip3 install --ignore-installed PyYAML==6.0

# Installing LRE packages now required for LOR as well as examples
sudo apt install -y liblre-cuda10 liblre-dev
