# LatentAI LRE - Detector Inference C++ example 
This folder contains a sample project for image classifier models.  This example supports the following:

- Model(s): LEIP Recipes Detectors
- DLDevice: CPU or CUDA
- LRE object: C++

## Quick Start

See the provided `inference_commands.bash` script.  This can be used as an example to run FP32, FP16 and INT8 versions of a model.  To use this script:

1. Install the device dependencies.  [Use the appropriate scripts for your device](../../setup_scripts)
2. Copy the modelLibrary.so to the device.
3. Edit the `inference_commands.bash` script to set `FLOAT32_MODEL` and `INT8_MODEL` variables to reflect your model paths. Specify the detector for pre and post processing setup. Supported are: YOLO, MOBNETSSD, EFFICIENTDET, NANODET
4. run `bash inference_commands.bash`

If you are only targeting C++, you may not wish to install everything in those setup scripts, but you may wish to use them for reference.
The critical dependencies for the C++ examples are listed below.


## Dependencies:

- OpenCV
- LatentAI Runtime Engine (LRE)
- Torch


## Building the Project Directly
If you would prefer to build and use the example application directly without the `inference_commands.bash` script, you will follow these basic steps:

Building the application:<br>
```
mkdir build
cd build
cmake ..
make -j 8
```

The generated binary will be placed in the *bin* folder with the name of **application**
you can then run this binary(inference) giving the following inputs for 10 test iterations  (We recommend running at least ten iterations if you are looking for accurate timing information):

```
path to binary               - bin/application
path to model                - modelLibrary.so
number of iterations         - 10
image to be evaluated        - ../../sample_images/penguin.jpg\n
model family                 - YOLO, MOBNETSSD, EFFICIENTDET, NANODET
```
<br>
The inference command for Float32 would be:

```
./bin/application \
  <path to>/model/Float32-compiled/modelLibrary.so \
  10 \
  ../../sample_images/penguin.jpg \
  YOLO
```
