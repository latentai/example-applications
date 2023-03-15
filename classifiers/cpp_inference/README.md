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
    ├── inference
    │   ├── images                 # Example images for running inference
    │   ├── model                  # Compiled(INT8 / FP32) models 
    │   └── labels                 # label files
    ├── test                       # Example Specs/Tests
    ├── application.cpp            # App (main) Implementation
    ├── CMakeLists.txt             # cMake for this cpp project
    ├── run_inference.run          # one-liner script for running inference
    └── README.md                  # This file.

1. Library prerequisites:
- TVM
- OpenCV
- Python3.6

2. Build project
 <code>mkdir build
cd build
cmake ..
make -j 8
</code>

## New Folder Structure
    .    
    ├── bin                        # Folder containing the binaries
    │   └── application           # Latent AI Runtime excecutable
    ├── cmake 
    ├── include                   
    ├── src                     
    ├── test                       
    ├── application.cpp  
    ├── CMakeLists.txt            
    ├── run_inference.run         
    └── README.md                  

The generated binary will be placed in the *bin* folder with the name of **application**
you can then run this binary(inference) giving the following inputs:

<code>path to LRELite binary      - bin/application
path to model                - modelLibrary.so
image to be evaluated        - inference/images/penguin.jpg\n
label names input            - inference/labels/class_names_10.txt\n
</code>

The inference command for Float32 would be:

```
./bin/application \
  inference/model/Float32-compiled/modelLibrary.so \
  inference/images/penguin.jpg \
  inference/labels/class_names_10.txt
```

## Example Output:

| Detections |
|:---------|
|The image prediction result is: id 6 Name: Penguin Score: 0.34167|
