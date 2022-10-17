# LatentAI LRE Lite - Classifier Inference C++ example 
This folder contains a sample project for usage of latentai.lre for:
- Model: timm-gernet_m 
- DLDevice: CPU /CUDA
- LRE object: C++

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
    │   └── latentai.lre           # Latent AI Runtime excecutable
    ├── cmake 
    ├── include                   
    ├── src                     
    ├── test                       
    ├── application.cpp  
    ├── CMakeLists.txt            
    ├── run_inference.run         
    └── README.md                  

The generated binary will be placed in the *bin* folder with the name of **latentai.lre**
you can then run this binary(inference) giving the following inputs:

<code>path to LRELite binary - bin/latentai.lre
path to model                - modelLibrary.so
image to be evaluated        - inference/images/penguin.jpg\n
label names input            - inference/labels/class_names_10.txt\n
</code>

The inference command would be:
./bin/latentai.lre inference/model/Float32-compiled/modelLibrary.so  inference/images/penguin.jpg inference/labels/class_names_10.txt 

## Example:

This single line is added in [run_inference.run](run_inference.run) (Don't forget running <code>chmod +x run_inference.run</code>):
| Detections |
|:---------|
|The image prediction result is: id 6 Name: Penguin Score: 0.34167|
