# LatentAI LRE - Detector Inference C++ example 
This folder contains a sample project for image detector models.  This example supports the following:

- Model(s): [LEIP Compiler Framework](https://leipdocs.latentai.io/cf/3.0/content/) compiled detectors
- DLDevice: CPU or CUDA
- LRE object: C++
- Model precisions: INT8, FP32, FP16
- One input image (e.g., JPG)

## Quick Start

Refer to the provided `inference_commands.bash` script. This can be used as an example to run FP32, FP16, and INT8 versions of a model. Run the following steps to use this script:

### 1. Make the Device Ready
Install the device dependencies. [Use the appropriate scripts for your device](../../setup_scripts)
### 2. Make the Model Available
Copy the modelLibrary.so to the device. Models compiled with the LEIP Compiler Framework have the following structure (e.g., model_name=efficentdet, architecture=aarch64_cuda_xavier_jp4):
```
<model_name>/
│
└── <architecture>/
    ├── Float32-compile/
    │   ├── modelLibrary.so
    │   ├── model_schema.yaml
    │   ├── processors/
    │   │   ├── af_preprocessor.json
    │   │   └── general_detection_postprocessor.py
    │   └── results.json
    ├── Int8-optimize/
    │   ├── modelLibrary.so
    │   ├── compression_report.html
    │   ├── model_schema.yaml
    │   ├── processors/
    │   │   ├── af_preprocessor.json
    │   │   └── general_detection_postprocessor.py
    │   └── results.json
    └── results.json
```

The bash script handles the running of the three supported precisions. This is done automatically if you set the MODEL_PATH:
```bash
MODEL_PATH=<path_to_model_name>
```
Or you can set the path to model compiled for FP32 or INT8 only.

### 3. Run for Three Precisions Using the inference_commands.bash Ccript
run  ``` bash inference_commands.bash --model_path </path/to/model> --img_path <path/of/image> --iterations <number of iterations> --model_family <model family> --conf_thres <confidence threshold> --iou_thres <iou threshold> ```
### 4. Example 
``` bash 
inference_commands.bash --model_path /workspace/yolov8/x86_64_cuda/ --img_path ../../sample_images/bus.jpg --iterations 100 --model_family YOLO --conf_thres 0.5 --iou_thres 0.45 
```

You may not need to install everything in those setup scripts if you are only targeting C++. However, you may wish to use the scripts for references.
The critical dependencies for the C++ examples are listed below.

### 5. Output
The script will provide a table showing the box coordinates, the score, and class for the detections. The annotated image will only contain boxes and the classes will not be printed in the image.
```bash
-----------------------------------------------------------
                     Box                   Score     Class
-----------------------------------------------------------
 207.5532  212.9904  282.8233  286.7948    0.8737    1.0000
 112.7287  345.0817  146.8491  377.6378    0.7854    1.0000
[ CUDAFloatType{2,6} ]
-----------------------------------------------------------
Write the annotated image to /home/dev/example_applications/sample_images/road314_March_01_2024_22:54:05_out.jpg
```
A JSON-like output containing the timings will be calculated:
```json
{
  “Average Inference Time ms”: {
    “Mean”: 5.886,
    “std_dev”: 0.102
  },
  “Average NMS Time ms”: {
    “Mean”: 1.01,
    “std_dev”: 0.113
  },
  “Average Preprocessing Time ms”: {
    “Mean”: 1.408,
    “std_dev”: 0.163
  },
  “Average Total Postprocessing Time ms”: {
    “Mean”: 2.505,
    “std_dev”: 0.238
  },
  “Precision”: “int8",
  “Total Time ms”: 9.799,
  “UID”: “4774ca59-c088-4bc7-9bf9-e13de920ce28”
}
```

## Dependencies:

- OpenCV
- LatentAI Runtime Engine (LRE)
- Torch
- Torchvision (optional; recommended for optimum post processing on GPU Targets)


## Building the Project Directly
The bash script handles the running of the three supported precisions. This is done automatically if you set the MODEL_PATH: The bash script handles the running of the three supported precisions. This is done automatically if you set the MODEL_PATH:If you would prefer to build and use the example application directly without the `inference_commands.bash` script

Building the application:<br>
```
mkdir build
cd build
cmake ..
make -j$(nproc)
```

The generated binary will be placed in the *bin* folder with the name of **application**. This binary(inference) can be run giving the following inputs for 100 test iterations. We recommend running at least 100 iterations if you are looking for accurate timing information:

```
path to binary               - bin/application
path to model                - modelLibrary.so
number of iterations         - 100
image to be evaluated        - ../../sample_images/penguin.jpg\n
model family                 - YOLO, MOBNETSSD, EFFICIENTDET, NANODET
confidence threshold         - 0.6
iou threshold                - 0.45
precision                    -float32,float16,int8   
```
<br>
For example, the inference command for Float32 will be:

```
bin/application --model_path /workspace/yolov5/x86_64_cuda/Float32-compile/modelLibrary.so 
--img_path ../../../sample_images/bus.jpg 
--iterations 100 
--model_family YOLO 
--iou_thres 0.45 
--conf_thres 0.6 
--precision float32
```
