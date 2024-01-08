#!/bin/bash

if [ ! -f /etc/apt/sources.list.d/latentai-stable.list ]; then
    echo "Error: Add latentai apt repo before running this script."
    echo ""
    echo "Example:"
    echo " wget -qO - https://public.latentai.io/add_apt_repository | sudo bash"
    echo " sudo apt update"
    exit
fi

sudo apt update
sudo apt install pylre
sudo apt install -y cmake
sudo apt install -y libopencv-dev
sudo apt install -y liblre-cpu liblre-dev

# Installed in the lor setup script, not required here
# sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

pip3 install opencv-python

