<img src=https://latentai.com/wp-content/uploads/2022/10/logo.svg width=300/><br />

# RELEASE NOTES

- 2.8
    - First edition of the example applications repo to be tracked in versions paralled to [LEIP SDK tool](https://docs.latentai.io/leip-sdk/v2.8/release-notes) 
    - Extended **- STEP 4.- Deployment -** from our [LEIP Recipes Overview](https://docs.latentai.io/leip-sdk/v2.8/leip-recipes-overview) 
    - Added C++ and python inference examples for **classifiers** and **detectors** (YoloV5, MobilenetSSD and EfficientDet)
    - Published the runtime shared object as a debian package to be used by the inference applications for: 
        - CPU Target  : **latentai-runtime-cpu**, 
        - CUDA Target : **latentai-runtime-cuda** and 
        - dev         : **latentai-runtime-dev** (containing also header files for runtime)
    - Examples have been tested in devices:
        - [Xavier AGX/NX](setup_scripts/agx_nx) (Jetpack 4.6.x)
        
            | Software           | Version  |
            | ------------------ | -------- |
            | latentai-runtime   | 2.8      |
            | Python             | 3.6      |
            | OpenCV             | 4.5.4.6  |
            | Typing Extensions  | 3.10.0.2 |
            | HuggingFace Hub    | 0.4.0    |
            | Effdet             | 0.2.4    |
            | PyYAML             | 6.0      |
            | Numpy              | 1.19.4   |
            | Protobuf           | 3.13.6   |
            | Pillow             | 8.4      |
            | CMake              | 3.10.2   |
            | Torch              | 1.10.0   |
            | Torchvision        | 0.11.3   |
        - [Raspberry Pi](setup_scripts/rpi) (64-bit Raspberry Pi OS)
            | Software           | Version  |
            | ------------------ | -------- |
            | latentai-runtime   | 2.8      |
            | Python             | 3.9      |
            | setuptools         | 4.5.4.6  |
            | CMake              | 3.10.2   |
            | PyTorch            | 1.13.0   |
            | Torchvision        | 0.14.0   |
