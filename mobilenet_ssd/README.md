# LatentAI LRE 

## C++ YOLOV5 Torch Example

## Prerequisites 

### LIBTORCH
#### Installation instructions for CPU Version
<code>wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip
</code>

#### Installation instructions for CUDA 11.6 Version
<code>wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu116.zip
</code>

#### Installation instructions for CUDA 11.7 Version
<code>wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip
</code>

#### Installation instructions for Jetpack
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
<code>wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
pip install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
</code> 

### TORCHVISION
<code>sudo apt install -y libjpeg-dev zlib1g-dev libavcodec-dev libpng-dev
git clone --branch v0.13.1 https://github.com/pytorch/vision torchvision
cd torchvision
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. #Add -DWITH_CUDA=on support for the CUDA if needed #For Jetpack 5.0.2 cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DWITH_CUDA=on .. 
make -j12
make install

</code>

### OpenCV
<code>apt install libopencv-dev
</code>

## Inputs
- modelLibrary.so: Generated using the LatentAI's SDK
- image_directory: Images to be processed

## Application Usage
<code>mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. #For Jetpack cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j8
bin/application path/to/modelLibrary.so path/to/images_drectory 
</code>
