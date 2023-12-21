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

- [Xavier AGX/NX](setup_scripts/agx_nx) (Jetpack 4.6.x and 5.0)
- [Raspberry Pi](setup_scripts/rpi) (64-bit Raspberry Pi OS)
- [LEIP Compiler Framework](setup_scripts/x86_cuda) (For use inside the LEIP Compiler Framework)

For specific instructions on installing the dependencies, see the `README.md` files in the script directories.

