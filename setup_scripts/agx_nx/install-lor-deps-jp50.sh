#!/bin/bash

TORCH_URL=https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl

# Check if nvcc is added to path
CUDA_PATH="/usr/local/cuda"
BASHRC_PATH="$HOME/.bashrc"
if [[ -d "$CUDA_PATH" ]]; then
  if ! grep -qF "export PATH=$CUDA_PATH/bin:~/.local/bin:\$PATH" "$BASHRC_PATH"; then
    echo "export PATH=$CUDA_PATH/bin:~/.local/bin:\$PATH" >> "$BASHRC_PATH"
    echo "Added CUDA path to $BASHRC_PATH. Please source $BASHRC_PATH rerun this script."
    exit
  else
    echo "CUDA & LOR already present in $BASHRC_PATH"
  fi
else
  echo "export PATH=~/.local/bin:\$PATH" >> "$BASHRC_PATH"
  echo "CUDA path not found, only LOR added"
fi

if [ ! -f /etc/apt/sources.list.d/latentai-stable.list ]; then
    echo "Info: Adding Latent AI repository..."
    wget -qO - https://public.latentai.io/add_apt_repository | sudo bash
fi

# Preload libraries needed for packages to initialize on Jetson
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip
pip3 install scikit-learn==1.2.2 numpy==1.23.5
echo "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:~/.local/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:\$LD_PRELOAD" >> "$BASHRC_PATH"

# OS dependencies
sudo apt install -y unzip
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

# Needed for LOR/Evaluate
sudo apt install -y libjpeg62 libjpeg8-dev

pip3 install Cython
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt install -y software-properties-common
sudo add-apt-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt install -y cmake

# Pytorch and Torchvision
sudo pip3 install $TORCH_URL

# Orin Torch+TorchVision hotfix for sm_87
HOTFIX_FILE=/usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
sudo sed -i "s/\('Ampere', '8\.0;8\.6+PTX\)/\1;8.7+PTX/" $HOTFIX_FILE
sudo sed -i "s/'8.6', '8.9', '9.0']/'8.6', '8.7', '8.9', '9.0']/" $HOTFIX_FILE

sudo apt install -y libopenblas-base libopenmpi-dev
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libpng-dev
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision
(cd torchvision || exit && sudo python3 setup.py install)

# Install Torchvision for c++
mkdir torchvision/build_cpp
cd torchvision/build_cpp
cmake \
    -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DWITH_CUDA=/usr/local/cuda ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..

# Needed for 2.9
pip3 install typing-extensions==3.10.0.2
pip3 install --ignore-installed PyYAML==6.0

# Install Latent AI LRE packages
sudo apt install -y liblre-cuda11 liblre-dev
sudo apt -y autoremove
