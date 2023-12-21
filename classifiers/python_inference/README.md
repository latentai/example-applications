# LatentAI LRE - Classifier Inference Python example 

This folder contains a python script that employs the LRE model to run an inference example. To run this inference example, do the following:

1. Install the dependencies on your device. [Use the appropriate scripts for your device](../../setup_scripts)
2. Install the LatentAI runtime engine (`pylre`) by `sudo apt install pylre` in the device.
3. Send the `FLOAT32_MODEL`, `INT8_MODEL` and `LABELS_PATH`, in that order to run [script](inference_commands.bash) `bash inference_commands.bash`. 

<br>
For example:

```
bash inference_commands.bash ~/models/timm-gernet_m/x86_64_cuda/Float32-compile ~/models/timm-gernet_m/x86_64_cuda/Int8-optimize ../../sample_images/apple.jpg
```