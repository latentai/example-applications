# LatentAI LRE C++ EfficientDet Example (with Torch)
This folder contains a sample project for efficientdet detection models.  This example supports the following:

- Model(s): LEIP Recipes - EfficientDet
- DLDevice: CPU or CUDA
- LRE object: C++

This variation of the C++ example does not require `torchvision` as a dependency.

### Quick Start
See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.  [Use the appropriate scripts for your device](../../../setup_scripts)
2. Copy the `Float32-compile` and `Int8-optimize` C++ LRE objects (`modelLibrary.so`) to the device
3. Edit the `inference_commands.bash` script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to reflect your model paths.
4. For CPU targets only - edit `application.cpp` and change {kDLCUDA, 0} to {kDLCPU,0}
5. run `bash inference_commands.bash`


If you are only targeting C++, you may not wish to install everything in the setup scripts, but you may wish to use them for reference. The critical dependencies for the C++ examples are listed below.

## Dependencies
- Libtorch
- OpenCV
- LatentAI Runtime Library

### Known Issues:

##### Libtorch: Jetpack / Torch 1.10.0 Known Issues

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




## Building the Project Directly
```
# Create build directory
mkdir build 
cd build

## Choose one of these two options
# cmake option 1 (default)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. 
# cmake option 2 (Jetpack)
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

# Make the application
make -j8
```
The generated binary will be placed in the bin folder with the name of application you can then run this binary(inference) giving the following inputs for 10 test iterations (We recommend running at least ten iterations if you are looking for accurate timing information):
```
modelLibrary.so: Model file generated using the LatentAI's SDK
<number>       : Number of Inference Iterations.  Use at least ten if seeking accurate timing information.
Image          : Image file to be processed
```
### Usage
bin/application path/to/modelLibrary.so number_of_iterations path/to/images_directory 
