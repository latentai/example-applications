# LatentAI LRE - Efficientdet Inference Python example 

This folder contains a python script that employs the LRE package to run an inference example. To run this inference example, do the following:

1. Copy the Python LRE objects from Float32-package and Int8-package to the device.
2. Install the `effdet` python package (see prerequisites below)
3. Edit `FLOAT32_PACKAGE` and `INT8_PACKAGE` in the bash script to point to the package directories.
4. Run `bash inference_commands.bash`


## Pre-requisites

The efficientdet package is used by the default pre/post processing in the Efficientdet family recipes, and needs to be installed before using these python examples.

**Jetpack 4.6 Installation (Python 3.6)**

If installing for Jetpack 4.6, we recommend you use the `install-lor-deps-jp46.sh` install script in the `setup_scripts/agx_nx` directory of this repo to install `effdet==0.2.4` and its dependencies.

**Python 3.8 / 3.9 Installation**
```
pip3 install effdet
```
The rest of the required packages have been precompiled and are part of the LRE package.
