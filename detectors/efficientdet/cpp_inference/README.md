# LatentAI LRE 

## C++ EfficientDet Torch/Torchvision Example

### Quick Start
See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.
2. Copy the `FLoat32-compile` and `Int8-optimize` C++ LRE objects `modelLibrary.so` to the device
3. Edit the inference_commands script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to point to your model paths.
4. For CPU targets change {kDLCUDA, 0} to {kDLCPU,0} in application.cpp
5. run `bash inference_commands.bash`

For step one,  we suggest starting with the provided setup scripts. [Please see the dependencies section of the top level README](../../README.md)

If you are only targeting C++, you may not wish to install everything in those setup scripts, but you may wish to use them for reference.

The critical dependencies for the C++ examples are listed below.

## Prerequisites 

### LIBTORCH
#### Installation instructions for CPU Version
    
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip


#### Installation instructions for CUDA 11.6 Version
    
    wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu116.zip


#### Installation instructions for CUDA 11.7 Version
    
    wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip

#### Installation instructions for Jetpack 4.6
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

    wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -o torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

##### Jetpack 4.6 / Torch 1.10.0 Known Issues

 **Compilation Error:**

```
error: ‘DLContext’ does not name a type; did you mean ‘Context’?  
    TORCH_API DLContext getDLContext(const Tensor& tensor, const int64_t& device_id);  
    ^~~~~~~~~
```
**Resolution for Compilation Error:**

  In line 17 of this file:
  
  `~/.local/lib/python3.6/site-packages/torch/include/ATen/DLConvertor.h`
```
// Change this line:
TORCH_API DLContext  getDLContext(const Tensor& tensor,  const  int64_t& device_id);

// To this:
TORCH_API DLDevice  getDLContext(const Tensor& tensor,  const  int64_t& device_id);
```
**Linking Error:**

```bin/application: error while loading shared libraries: libc10_cuda.so: cannot open shared object file: No such file or directory```
  
**Resolution for Linking Error:**

Add torch libraries to LD_LIBRARY_PATH:

```
export LD_LIBRARY_PATH=~/.local/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH
```

  

#### Installation instructions for Jetpack 5.0
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

    wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
    pip install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
 

### TORCHVISION
    sudo apt install -y libjpeg-dev zlib1g-dev libavcodec-dev libpng-dev
    git clone --branch v0.13.1 https://github.com/pytorch/vision torchvision
    cd torchvision
    mkdir build 
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. 
    #Add -DWITH_CUDA=on support for the CUDA if needed #For Jetpack 5.0.2 cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DWITH_CUDA=on .. 
    make -j12
    make install

### OpenCV
    apt install libopencv-dev

### LatentAI Runtime Libraries
```
# Add the Latent AI debian repository to your apt lists
sudo sh ../../../setup_scripts/add_latentai_debian_repository.sh

# For CPU Target, install the cpu runtime
# sudo apt install latentai-runtime-cpu

# For CUDA Target, install the gpu runtime
sudo apt install latentai-runtime-cuda

# You will also need to install the runtime development package
sudo apt install latentai-runtime-dev
```

## Inputs
- modelLibrary.so: Generated using the LatentAI's SDK
- image_directory: Images to be processed

## Application Usage
    mkdir build 
    cd build
    
    # cmake option 1 (Default):
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. 
    # cmake option 2 (Jetpack):
    cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

    make -j8

    # Usage:
    bin/application path/to/modelLibrary.so path/to/images_directory 
