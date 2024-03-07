# LatentAI LRE - Classifier Inference C++ example 
This folder contains a sample project for image classifier models.  This example supports the following:

- Model(s): [LEIP Compiler Framework](https://leipdocs.latentai.io/cf/3.0/content/) compiled classifiers
- DLDevice: CPU or CUDA
- LRE object: C++
- Model precisions: INT8, FP32, FP16
- One input image (e.g., JPG)

## Quick Start

Refer to the provided `inference_commands.bash` script. This can be used as an example to run FP32, FP16, and INT8 versions of a model. Run the following steps to use this script:

1. Install the device dependencies. [Use the appropriate scripts for your device.](../../setup_scripts)
2. Copy the modelLibrary.so to the device.
3. Run  ``` bash inference_commands.bash --model_path </path/to/model> --img_path <path/of/image> --iterations <number of iterations> -label_file <path/to/labels> ```
4. Example ``` bash inference_commands.bash --model_path /workspace/classifier/x86_64_cuda/ --img_path ../../sample_images/bus.jpg --iterations 100 --label_file ../../labels/class_names_10.txt ```
5. The script will provide a JSON-like output and will provide the following information:
``` json
{
    "UID": "032ecf56-2389-495f-9740-af43c3c0ef68",
    "Precision": "float32",
    "Device": "DLDeviceType::kDLCUDA",
    "Input Image Size": [
        1024,
        997,
        3
    ],
    "Model Input Shapes": [
        [
            1,
            3,
            224,
            224
        ]
    ],
    "Model Input Layouts": [
        "NCHW"
    ],
    "Average Preprocessing Time ms": {
        "Mean": 4.585,
        "std_dev": 0.431
    },
    "Average Inference Time ms": {
        "Mean": 2.086,
        "std_dev": 0.066
    },
    "Average Total Postprocessing Time ms": {
        "Mean": 0.307,
        "std_dev": 0.141
    },
    "Total Time ms": 6.978,
    "Class": "Apple",
    "Score": 0.8374396562576294
}
```

You may not need to install everything in those setup scripts if you are only targeting C++. However, you may wish to use the scripts for references.
The critical dependencies for the C++ examples are listed below.


## Dependencies:

- OpenCV
- LatentAI Runtime Engine (LRE)


## Building the Project Directly
Follow these basic steps if you would prefer to build and use the example application directly without the `inference_commands.bash` script:

Building the application:<br>
```
mkdir build
cd build
cmake ..
make -j 8
```

The generated binary will be placed in the *bin* folder with the name of **application**. You can then run this binary (inference) giving the following inputs for 10 test iterations. We recommend running at least 10 iterations if you are looking for accurate timing information:


```
path to binary               - bin/application
path to model                - modelLibrary.so
number of iterations         - 100
image to be evaluated        - ../../sample_images/penguin.jpg\n
label names input            - ../../labels/class_names_10.txt\n
precision                    -float32,float16,int8   
```
<br>
For example, the inference command for Float16 will be:

```
bin/application --model_path /workspace/yolov5/x86_64_cuda/Float32-compile/modelLibrary.so 
--img_path ../../../sample_images/bus.jpg 
--iterations 100 
--iou_thres 0.45 
--conf_thres 0.6 
--precision float16
--label_file ../../labels/class_names_10.txt
```
