# LatentAI LRE 

## C++ EfficientDet Torch/Torchvision Example

### Quick Start
See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.
2. Copy the `Float32-compile` and `Int8-optimize` C++ LRE objects `modelLibrary.so` to the device
3. Edit the inference_commands script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to point to your model paths.
4. For CPU targets change {kDLCUDA, 0} to {kDLCPU,0} in application.cpp
5. run `bash inference_commands.bash`

For step one,  we suggest starting with the provided setup scripts. [Please see the dependencies section of the top level README](../../../README.md)

If you are only targeting C++, you may not wish to install everything in those setup scripts, but you may wish to use them for reference.

The critical dependencies for the C++ examples are listed below.

## Prerequisites 

### LIBTORCH

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

### TORCHVISION

### OpenCV

### LatentAI Runtime Library

## Inputs
- modelLibrary.so: Generated using the LatentAI's SDK
- number : Number of Inference Iteration.
- Image: Image to be processed

## Application Usage
    mkdir build 
    cd build

    # cmake option 1 (default)
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. 
    # cmake option 2 (Jetpack)
    cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

    make -j8
    
    # Usage
    bin/application path/to/modelLibrary.so number_of_iterations path/to/images_directory 

