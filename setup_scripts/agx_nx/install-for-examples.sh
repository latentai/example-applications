#!/bin/bash

sudo bash ../add_latentai_debian_repository.sh
sudo apt update

sudo apt install -y libopencv-dev
sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo apt install -y liblre-cuda liblre-dev

pip3 install scikit-build
#pip3 install opencv-python
pip3 install opencv-python==4.5.4.60
pip3 install gdown

