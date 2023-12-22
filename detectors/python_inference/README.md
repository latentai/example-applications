# LatentAI LRE - Detector Inference Python example 

This folder contains a python script that employs the LRE model to run an inference example. To run this inference example, do the following:

1. Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)
2. Install the LatentAI runtime engine (`pylre`) by `sudo apt install pylre` in the device.
3. Send the arguments
```
model_binary_path => path to compiled model binary
input_image_path  => path to input image
labels            => path to labels
model_format      => type of detector (efficientdet, ssd, yolo, nanodet)
max_det           => maximum detections per image
confidence        => confidence treshold (e.g 0.3)
iou               => iou treshold (e.g. 0.4)
```
to run [script](infer.py) `python3 infer.py`. 

<br>
For example, the inference command for Float32 would be:

```
python3 infer.py \
  --model_binary_path <path to model>/Float32-compiled/modelLibrary.so \
  --input_image_path../../sample_images/penguin.jpg \
  --labels ../../labels/class_names_10.txt
  --model_format <model format: efficientdet/ssd/yolo/nanodet> 
  --max_det <integer of max detections> 
  --confidence <float of confidence threshold >
  --iou <float of IOU threshold>
```

There is a helpful [script](inference_commands.bash) to run all compiled artifacts from a leip pipeline.