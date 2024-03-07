# LatentAI LRE - Detector Inference Python example 

This folder contains a python script that employs the LRE (LatentAI Runtime Engine) to run an inference example. It supports:

- Model(s): [LEIP Compiler Framework](https://leipdocs.latentai.io/cf/3.0/content/) compiled detectors
- DLDevice: CPU or CUDA
- LRE object: C++
- Model precisions: INT8, FP32, FP16
- One input image (e.g., jpg)

Perfom the following steps to run this example:

### 1. Make the device ready
Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)

### 2. Make the model available 
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

The bash script handles running the three supported precisions, automatically if you set the MODEL_PATH:
```bash
MODEL_PATH=<path_to_model_name>
```
or you can set the path to model compiled for FP32 or INT8 only.


### 3. Run 
The [script](infer.py) `python3 infer.py`, can be run from the command line with the given parameters. 

  | Parameter                   |          | Description                                          |
  |------------------------     | ---------|------------------------------------------------------|
  | --compiled_model_path       | Required |Path to the directory containing INT8 and FP32 models. If only one of the models is to be run, set the float32_model_binary_path or int8_model_binary_path  parameter instead|
  | --float32_model_binary_path | Optional | Path to the Float32 model binary file.               |
  | --int8_model_binary_path    | Optional | Path to the Int8 model binary file.                  |
  | --input_image_path          | Required |Path to the image file for inference.                 |
  | --iterations                | Required |Number of iterations for inference.                   |
  | --model_family              | Required |Type of model family (supported detectors: YOLO, MOBNETSSD, EFFICIENTDET, NANODET).                                                                                       |
  | --confidence_threshold      | Required |Confidence threshold for inference.                   |
  | --iou_threshold             | Required |IoU threshold for inference (Intersection over Union).|
  | --labels_path               | Required |Path to the labels file.                              |
  | --maximum_detections        | Optional |Maximum number of detections that will be selected.   |

<br>
```
python3 infer.py \
  --float32_model_binary_path <path to model>/Float32-compiled/modelLibrary.so \
  --input_image_path../../sample_images/penguin.jpg \
  --iterations 100 \
  --model_family <model format: efficientdet/ssd/yolo/nanodet> 
  --confidence_threshold <float of confidence threshold>
  --iou_threshold <float of IOU threshold>
  --labels_path ../../labels/class_names_10.txt
  --maximum_detections <integer of max detections> 
```

### 4. Run for three precisions using the inference_commands.bash script
There is a helpful [script](inference_commands.bash) to run all compiled artifacts from a leip pipeline.
This will run ONE model (e.g. efficientdet) at the three different supported model precisions.
To run it simply do:
<br>
```
python3 infer.py \
  --compiled_model_path <path to models> \
  --input_image_path../../sample_images/penguin.jpg \
  --iterations 100 \
  --model_family <model format: efficientdet/ssd/yolo/nanodet> 
  --confidence_threshold <float of confidence threshold>
  --iou_threshold <float of IOU threshold>
  --labels_path ../../labels/class_names_10.txt
  --maximum_detections <integer of max detections> 
```
---

### 5. Output
Currently the script will provide a json-like output for the detections. The annotated image will contain boxes and labels.
```json
{
  "UUID": "2bb5b34a-b8a2-45cb-ad59-bf1a7c22e0a2",
  "Precision": "int8",
  "Device": "DLDeviceType::kDLCUDA",
  "Input Image Size": [
    1080,
    810,
    3
  ],
  "Model Input Shapes": [
    [
      1,
      3,
      640,
      640
    ]
  ],
  "Model Input Layouts": [
    "NCHW"
  ],
  "Average Preprocessing Time ms": {
    "Mean": 14.702,
    "std_dev": 0.81
  },
  "Average Inference Time ms": {
    "Mean": 3.681,
    "std_dev": 0.018
  },
  "Average Total Postprocessing Time ms": {
    "Mean": 2.514,
    "std_dev": 0.341
  },
  "Total Time ms": 20.897,
  "Annotated Image": "../../sample_images/bus-2024-02-27 06:12:14.130030.jpg"
}
```
# Doing your own python inference script using pylre
## **Invoking the Python LRE**
After you have successfully compiled a model from [LEIP Compiler Framework](https://leipdocs.latentai.io/cf/3.0/content/) and installed pylre on your deployment environment, you can load your compiled artifact as follows,

```bash
from pylre import LatentRuntimeEngine
lre = LatentRuntimeEngine("modelLibrary.so")
```

Once you have loaded an lre, you can use lre APIs for configuration, inference, and introspection.
For example you can use,

```bash
lre.get_metadata()
```

which allows you to visualize critical information about your compiled model such as:

**lre.model_precision**: what precision is the model compiled with, especially relevant when quantizing.<br>
**lre.device_type**: what target device is the model compiled for, if the architecture of your target hardware does not match with what it is compiled for, then you will have to recompile the model to your target hardware.<br>
**lre.input_shapes**: what input shapes does the model expect, if the input images are not compatible with this shape, you'll get unexpected results.<br>
**lre.input_layouts**: what data layout has the model been compiled with, depending on your hardware, this could affect how your memory hierarchy is leveraged, which can have major performance implications.<br>

When you're satisfied with the loaded model, you can generate/ format data to feed to the model and call the model as follows,

```bash
# input data can be a tensor of numpy, torch, or DLTensor.
# input data should match the shape, layout, and type of the model input.
lre.infer(input_data)
```

which will run the model on the compiled target. You can either use lre API to access the outputs or merely pipe the output of the above expression as the output.

```bash
# access output with lre API
outputs = lre.get_outputs()

# pipe return of inference as the output
outputs = lre.infer(input_data)

# if you want to access a specific index of the output you can either
# index the tensor above as
outputs[0]

# or access only the relevant index as
lre.get_output(0)
```

The output you get will be a DLTensor. Most of the time, you'd want to convert this to a more amenable tensor library such as

```bash
# numpy (also useful if you're using cv2 for further computations)
numpy_tensor = np.from_dlpack(outputs[0])

# torch
torch_tensor = T.utils.dlpack.from_dlpack(outputs[0])
```

### Pre- and post- processing configurations

These example applications use several external libraries, which offer a tradeoff between performance, ease of design, ease of model swapping, target support.

1. **Albumnetations**
If you have generated your model from LEIP Recipe Designer, you may already have artifacts for pre-processing. Albumentations flow automatically detects this artifact and runs the pre-process transforms.

2. **Torch**
Pytorch provides easy to use transforms that can be easily run with better performance on CUDA targets. Torch based pre- and post-processing provides examples on how to write these transforms customized for some default LEIP recipes.

3. **OpenCV**
Some deployment targets may not have access to pytorch, or may not simply benefit from torch. In these cases, OpenCV is generally a more widely supported library to carry out transforms. CV2 based pre- and post- processing provides examples on how to write these transforms customized for some default LEIP recipes. These transforms are not optimized for performance.

Within our examples, we use several libraries for different tasks,
- pre-processing (albumentations/ torch/ opencv)
- post-processing (torch/ opencv)
- visualization (PIL/ opencv)

```bash
from utils import utils
# boolean albumentations: if true automatically targets albumentations transforms available in the environment, visualization is cv2, preprocess is albumentations.
# boolean visualization: if false selects PIL, compatible with torch; otherwise cv2 is compatible with others.
# boolean preprocess: if true torch, else cv2
# boolean postprocess: if true torch, else cv2
config = utils.Processors(albumentations, visualization, preprocess, postprocess)
config.display_config()
```

### Performance measurements

Latent AI example applications provide some infrastructure to do performance measurements.

```bash
from utils import utils
timer = utils.Timer()

timer.start()
lre.infer(input_data)
timer.stop()

time_in_ms = timer.averageElapsedMilliseconds()
```

This measures the execution time of an lre inference.

However, there are couple of things you can do to improve your inference time based on which target you're running.

- keep data in accelerator during inference, data transfer between compute devices can have a major impact
- for TRT, pre-build your execution engines before critical deployment
- for TRT, warming-up your embedded GPU device (memory loading, cache warm-up) will give you better inference time at steady state
- for TRT, when using torch post-processing, running post-processing once before steady state will improve better post-processing time
- for TRT, make sure you use a suitable profile based on your inference speed vs. energy tradeoff
- for TRT, if you're building TRT engines for multiple models set `LAI_TENSORRT_TIMING_CACHE` which can speedup engine build time
- set your confidence threshold, iou threshold, and max detections stricly to cut down amount of computations done during post-processing. However, this has to be carefully set to get a good tradeoff between generating useful detections, accuracy, and inference time.
- consider using a quantized model, depending on your target you can use compile time/ runtime configurations and floating point (32 bit/ 16 bit)/ integer precision.

