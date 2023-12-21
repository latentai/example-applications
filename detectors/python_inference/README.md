# LatentAI LRE - Detector Inference Python example 

This folder contains a python script that employs the LRE model to run an inference example. To run this inference example, do the following:

1. Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)
2. Install the LatentAI runtime engine (`pylre`) by `sudo apt install pylre` in the device.
3. Send the arguments
```
FLOAT32_MODEL   => path to model compiled for FP32
INT8_MODEL      => path to model optimized for INT8
LABELS_PATH     => path to labels
MODEL_FORMAT    => type of detector (YOLO, MOBNETSSD, EFFICIENTDET, NANODET)
MAX_DET         => maximum detections per image
CONF_THRESHOLD  => confidence treshold (e.g 0.3)
IOU_THRESHOLD   => iou treshold (e.g. 0.4)
```
, in that order to run [script](inference_commands.bash) `bash inference_commands.bash`. 

<br>
For example:

```
bash inference_commands.bash ~/models/ssdn/x86_64_cuda/Float32-compile ~//models/ssdn/x86_64_cuda/Int8-optimize ../../labels/pascal_voc.txt MOBNETSSD 5 0.3 0.45
```