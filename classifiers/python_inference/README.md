# LatentAI LRE - Classifier Inference Python example 

This folder contains a Python script that employs the LRE model to run an inference example. Perform the following steps to run this inference example: 

1. Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)
2. Install the LatentAI Runtime Engine (`pylre`) by `sudo apt install pylre` in the device.
3. Send the `model_binary_path`, `input_image_path` and `labels` to run [script](infer.py) `python3 infer.py`. 
5. Output
Currently the script will provide a JSON-like output providing the following information:
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

<br>
For example, the inference command for Float32 will be:

```
python3 infer.py \
  --model_binary_path <path to model>/Float32-compiled/modelLibrary.so \
  --input_image_path../../sample_images/penguin.jpg \
  --labels ../../labels/class_names_10.txt
```

There is a helpful [script](inference_commands.bash) that can be run by providing the `FLOAT32_MODEL` and `INT8_MODEL`, in that order.

<br>
For example:

```
bash inference_commands.bash \
  <path to>/model/Float32-compiled/modelLibrary.so \
  <path to>/model/Int8-optimized/modelLibrary.so
```
