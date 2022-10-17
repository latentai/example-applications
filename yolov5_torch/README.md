# LatentAI LRE 

## C++ YOLOV5 Torch Example

## Prerequisites 

### LIBTORCH
<code>wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip
</code>

### TORCHVISION
<code>sudo apt install -y libjpeg-dev zlib1g-dev libavcodec-dev libpng-dev
git clone --branch v0.13.1 https://github.com/pytorch/vision torchvision
cd torchvision
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j12
</code>

## Inputs
- modelLibrary.so: Generated using the LatentAI's SDK
- image_directory: Images to be processed

## Application Usage
<code>mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ ..
make -j8
bin/application path/to/modelLibrary.so path/to/images_drectory 
</code>
