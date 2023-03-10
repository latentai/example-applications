#!/bin/bash

sudo bash ../add_latentai_debian_repository.sh
sudo apt update

# cmake available for 18.04 via apt is too old
sudo apt remove -y --purge cmake
hash -r
sudo snap install cmake --classic
hash -r

sudo apt install -y libopencv-dev
sudo apt install -y latentai-runtime-cuda latentai-runtime-dev
sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

pip3 install scikit-build
pip3 install opencv-python
pip3 install gdown

