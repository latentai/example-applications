<img src=https://latentai.com/wp-content/uploads/2022/10/logo.svg width=300/><br />

# Example Applications for Model Deployment

[Developer Resources](https://docs.latentai.io) |
[LEIP Recipes](https://docs.latentai.io/leip-recipes/) |
[Request a Demo](https://latentai.com/contact-us/) |

Congratulations, you have a compiled model! Now it is time to deploy it in your target device.

In this public repository, the example applications (in both c++ and python) for the LatentAI recipe models are presented.
Specific instructions can be found under:

- Classifiers     [C++](classifiers/cpp_inference/README.md) | [python](classifiers/python_inference/README.md)
- Detectors
    - YoloV5           [C++](detectors/yolov5/cpp_inference/README.md) |  [C++ no torchvision](detectors/yolov5/cpp_inference_no_torchvision/README.md) | [python](detectors/yolov5/python_inference/README.md)
    - MobilenetSSD     [C++](detectors/mobilenet_ssd/cpp_inference/README.md) | [python](detectors/mobilenet_ssd/python_inference/README.md)
    - EfficientDet     [C++](detectors/efficientdet/cpp_inference/README.md) | [C++ no torchvision](detectors/efficientdet/cpp_inference_no_torchvision/README.md) | [python](detectors/efficientdet/python_inference/README.md)

To run the application, there are bash files to be run inside of each inference example folder.
For those scripts to work, replace the model path in the scripts or be sure to have saved the model in the same location.

You do not need to clone the whole repository but the [sample_images](sample_images/) and the [labels](labels/) directories are used by the scripts.

Good luck!

### Dependencies

To assist you in setting up your edge device to work with these examples, we have provided setup script for the following devices.  These scripts assume you are starting from a fresh OS install.

- [Xavier AGX/NX](setup_scripts/agx_nx) (Jetpack 4.6.x)
- [Raspberry Pi](setup_scripts/rpi) (64-bit Raspberry Pi OS)

To install a fresh version of Jetpack 4.6 on your Xavier device, refer to the [Nvidia Jetson documentation](https://developer.download.nvidia.com/embedded/L4T/r32-3-1_Release_v1.0/jetson_agx_xavier_developer_kit_user_guide.pdf)

Within the setup script directories, you will find scripts to install the LOR dependencies and to install for the examples.  _The example install script assumes that you have already run the setup script for the LOR_.

Depending on your needs, you may be able to avoid installing some of the dependencies for production deployment.  In particular, these scripts install a number of Python dependencies that would not be required for deployment of a C++ application and model. The scripts also install dependencies that relate to the recipe pre- and post- processing, and may not match your needs when you deploy your own processing code.  The packages supplied provide a good starting place during testing, and permit any of the Recipe models to work with the LOR and example applications for Python and C++.

### C++ Applications
-------------------
### Dependencies

At a minimum, to build your C++ Application, you will need to install the LEIP runtime packages.  These packages are installed automatically using the above setup scripts, but if you are planning to do a minimal installation you may install these packages using our debian reposistory:


    # Add the Latent AI debian repository to your apt lists
    sudo sh ./setup_scripts/add_latentai_debian_repository.sh

    # For CPU Target, install the cpu runtime
    # sudo apt install liblre-cpu

    # For CUDA Target, install the gpu runtime
    sudo apt install liblre-cuda

    # You will also need to install the runtime development package
    sudo apt install liblre-dev

Note: `liblre-cuda` expects cuda, cudnn and tensorrt packages to be installed


### CPU Targets
---------------
By default, in application.cpp the device type is set to CUDA. For CPU targets change {kDLCUDA, 0} to {kDLCPU,0}.


### Advanced CUDA Target Settings

 - To run at FP16 precision set environment variable
   `TVM_TENSORRT_USE_FP16=1`. 
 - To run at Int8 precision, set the variables    `TVM_TENSORRT_USE_INT8=1`  and `TRT_INT8_PATH=/path/to/Int8-optimize/.activations/`.  Requires optimized package from LEIP SDK.
 - Runtime Engines can be cached using `TVM_TENSORRT_CACHE_DIR=/path/to/store` . **Warning ! engines for different models should not be stored in the same directory.**

### Using LOR
-------------
It is possible to test a recipe model on the target device from the host SDK container using the LEIP Object Runner (LOR).  For more information on usiing the LOR, refer to the [LEIP Recipes Documentation](https://docs.latentai.io/)


