# LatentAI LRE 

## C++ YOLOV5 Torch Example

### Quick Start
See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.
2. Copy the `FLoat32-compile` and `Int8-optimize` C++ LRE objects `modelLibrary.so` to the device
3. Edit the inference_commands script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to point to your model paths.
4. For CPU targets change {kDLCUDA, 0} to {kDLCPU,0} in application.cpp
5. run `bash inference_commands.bash`


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


### OpenCV
    apt install libopencv-dev


## Inputs
- modelLibrary.so: Generated using the LatentAI's SDK
- image_directory: Images to be processed

## Application Usage
    mkdir build 
    cd build

    # cmake option 1 (default)
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. 
    # cmake option 2 (Jetpack)
    For Jetpack cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
    
    make -j8

    # Usage:
    bin/application path/to/modelLibrary.so path/to/images_directory 
