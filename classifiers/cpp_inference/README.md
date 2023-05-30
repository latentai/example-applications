# LatentAI LRE - Classifier Inference C++ example 
This folder contains a sample project for image classifier models.  This example supports the following:

- Model(s): LEIP Recipes Classifiers
- DLDevice: CPU or CUDA
- LRE object: C++

## Quick Start

See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.
2. Copy the modelLibrary.so C++ LRE object to the device
3. Edit the inference_commands script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to your model paths.
4. For CPU targets change `{kDLCUDA, 0}` to `{kDLCPU,0}` in `application.cpp`
5. run `bash inference_commands.bash`

For step one,  we suggest starting with the provided setup scripts. [Please see the dependencies section of the top level README](../../README.md)

If you are only targeting C++, you may not wish to install everything in those setup scripts, but you may wish to use them for reference.
The critical dependencies for the C++ examples are listed below.


# Example - User Manual
## Folder Structure
    .
    ├── cmake                   
    │   └── CppFlags.cmake         # Contains flags specific for the target system
    ├── include                   
    │   ├── lre_model.hpp                       # Propietary run time includes
    │   ├── imagenet_torch_nchw_processors.hpp  # Propietary pre and post processors 
    │   └── ...                                 # On-Demand includes.
    ├── src                     
    │   ├── lre_model.cpp                       # Propietary run time implementation
    │   ├── imagenet_torch_nchw_processors.cpp  # Propietary pre and post processors implementation 
    │   └── ...                                 # On-Demand includes.
    ├── application.cpp            # App (main) Implementation
    ├── CMakeLists.txt             # cMake for this cpp project
    ├── run_inference.run          # one-liner script for running inference
    └── README.md                  # This file.

## Dependencies:

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


## Build project
 <code>mkdir build
cd build
cmake ..
make -j 8
</code>

## New Folder Structure
    .    
    ├── bin                        # Folder containing the binaries
    │   └── application            # Latent AI Runtime excecutable
    ├── cmake 
    ├── include                   
    ├── src                                           
    ├── application.cpp  
    ├── CMakeLists.txt            
    ├── inference_commands.bash         
    └── README.md                  

The generated binary will be placed in the *bin* folder with the name of **application**
you can then run this binary(inference) giving the following inputs:

<code>path to LRELite binary      - bin/application
path to model                - modelLibrary.so
image to be evaluated        - ../../sample_images/penguin.jpg\n
label names input            - ../../labels/class_names_10.txt\n
</code>

The inference command for Float32 would be:

```
./bin/application \
  <path to>/model/Float32-compiled/modelLibrary.so \
  10 \
  ../../sample_images/penguin.jpg \
  ../../labels/class_names_10.txt
```

## Example Output:

| Detections |
|:---------|
|The image prediction result is: id 6 Name: Penguin Score: 0.34167|
