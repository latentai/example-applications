# LatentAI LRE - Classifier Inference Python example 

This folder contains a python script that employs the LRE model to run an inference example. To run this inference example, do the following:

1. Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)
2. Install the LatentAI runtime engine (`pylre`) by `sudo apt install pylre` in the device.
3. Send the `model_binary_path`, `input_image_path` and `labels` to run [script](infer.py) `python3 infer.py`. 

<br>
For example, the inference command for Float32 would be:

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