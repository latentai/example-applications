<img src=https://latentai.com/wp-content/uploads/2022/10/logo.svg width=300/><br />

# RELEASE NOTES

- [2.8](https://github.com/latentai/example-applications/blob/2.8/RELEASE_NOTES.md)
- [2.9](https://github.com/latentai/example-applications/blob/2.9/RELEASE_NOTES.md)
- 3.0
   - Latent AI Runtime Environment library names and debian packages have changed.  New LRE library packages:
        - CPU Target  : **liblre-cpu**, 
        - CUDA Target : **liblre-cuda10** or **liblre-cuda11** and 
        - Dev         : **liblre-dev** (containing also header files for runtime)
    - C++ Applications have been updated to be more consistent and with better performant Pre- and Post- Processing
    - C++ non torchvision detectors folder has been removed and it is now handled by [CMake](detectors/cpp_inference/CMakeLists.txt). Torchvision is used for NMS
    - LRE API updates:
        - Updated the API for INT8 inference (ENV_VARS setting is no longer needed as these information is part of the metadata)
        - Setting Precision is now possible as:

        |                          | Precision|@runtime              ||
        |--------------------------|---------|-----------|------------|
        | Precision @Compilation   | INT8    |   FP16    | FP32       |
        |     INT8                 |   ✔️     |     ✔️     |   ✔️        |
        |     FP16                 |         |     ✔️     |   ✔️        |
        |     FP32                 |         |     ✔️     |   ✔️        |

    - Examples have been tested against the following devices/dependencies:
        - [Xavier AGX/NX](setup_scripts/agx_nx) (Jetpack 4.6.x)
            | Software           | Version  |
            | ------------------ | -------- |
            | liblre-cuda10      | 3.0.0-1  |
            | pylre              | 3.0.0-1  |
            | Python             | 3.6      |
            | OpenCV             | 4.5.4.6  |
            | Typing Extensions  | 3.10.0.2 |
            | HuggingFace Hub    | 0.4.0    |
            | Effdet             | 0.2.4    |
            | PyYAML             | 6.0      |
            | Numpy              | 1.19.4   |
            | Protobuf           | 3.19.6   |
            | Pillow             | 8.4      |
            | CMake              | 3.26.4   |
            | Torch              | 1.10.0   |
            | Torchvision        | 0.11.3   |
        - [Orin AGX/NX](setup_scripts/agx_nx) (Jetpack 5.0.x)
            | Software           | Version  |
            | ------------------ | -------- |
            | liblre-cuda11      | 3.0.0-1  |
            | pylre              | 3.0.0-1  |
            | Python             | 3.8      |
            | OpenCV             | 4.5.4.6  |
            | Typing Extensions  | 3.10.0.2 |
            | HuggingFace Hub    | 0.4.0    |
            | Effdet             | 0.2.4    |
            | PyYAML             | 6.0      |
            | Numpy              | 1.23.5   |
            | Protobuf           | 3.20.3   |
            | Pillow             | 7.0.0    |
            | CMake              | 3.25.2   |
            | Torch              | 2.0.0    |
            | Torchvision        | 0.15.1   |
        - [Raspberry Pi](setup_scripts/rpi) (64-bit Raspberry Pi OS)
            | Software           | Version  |
            | ------------------ | -------- |
            | liblre-cpu         | 3.0.0-1  |
            | pylre              | 3.0.0-1  |
            | Python             | 3.9      |
            | setuptools         | 67.8.0   |
            | CMake              | 3.18.4   |
            | PyTorch            | 1.13.0   |
            | Torchvision        | 0.14.0   |
