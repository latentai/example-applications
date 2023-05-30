#!/bin/bash

bash ../add_latentai_debian_repository.sh
apt update

apt install -y libopencv-dev
apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
apt install -y liblre-cuda liblre-dev


wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip 

apt install -y libjpeg-dev zlib1g-dev libavcodec-dev libpng-dev
git clone --branch v0.13.1 https://github.com/pytorch/vision torchvision
cd torchvision
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DWITH_CUDA=ON .. 
make -j12
make install
ldconfig