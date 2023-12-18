# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python
import os
import torch as T
import torchvision.transforms as transforms
from PIL import Image

def main():
    from argparse import ArgumentParser
    from pathlib import Path

    from pylre import LatentRuntimeEngine

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--model_binary_path", type=str, default=".", help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="../../sample_images/apple.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Path to labels text file.",
    )

    args = parser.parse_args()

    # Load runtime
    lre = LatentRuntimeEngine(str(Path(args.model_binary_path) / "modelLibrary.so"))
    print(lre.get_metadata())

    # Set precision
    use_fp16 = bool(int(os.getenv("TVM_TENSORRT_USE_FP16", 0)))
    if(use_fp16):
        lre.set_model_precision("float16")
    
    # Read metadata from runtime
    layout_shapes = get_layout_dims(lre.input_layouts, lre.input_shapes)
    input_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))
    device = lre.device_type

    # Load image
    input_image_path = args.input_image_path
    image = Image.open(input_image_path)
    
    # Apply preprocess transformations
    resize_transform = transforms.Resize(input_size)
    resized_image = resize_transform(image)
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resized_image_normalized = normalize_transform(resized_image)

    # Run inference
    lre.infer(resized_image_normalized)
    
    # Post-process    
    outputs = lre.get_outputs()

    # Visualize
    output = outputs[0] 
    output = T.from_dlpack(output)
    op = postprocess_top_one(output)
    print_top_one(op, args.labels)


def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")
    
def postprocess_top_one(values):
    values = T.nn.functional.softmax(values, dim=1)
    max_index = T.argmax(values).item()
    max_value = values[0][max_index]        
    
    top_one = (max_index, max_value )
    return top_one

def print_top_one(top_one, label_file_name):
    with open(label_file_name, 'r') as label_file:
        lines = label_file.readlines()

    if top_one[0] >= 0 and top_one[0] < len(lines):
        label = lines[int(top_one[0])].strip()
    else:
        label = "Unknown Label"

    print(" ------------------------------------------------------------ ")
    print(" Detections ")
    print(" ------------------------------------------------------------ ")
    print(f" The image prediction result is: id {top_one[0]}")
    print(f" Name: {label}")
    print(f" Score: {top_one[1]}")
    print(" ------------------------------------------------------------ ")
    
def get_layout_dims(layout_list, shape_list):
    if len(layout_list) != len(shape_list):
        raise ValueError("Both input lists should have the same number of elements.")
    
    result = []
    
    for i in range(len(layout_list)):
        layout_str = layout_list[i]
        shape_tuple = shape_list[i]
        
        if len(layout_str) != len(shape_tuple):
            raise ValueError(f"Length of layout string does not match the number of elements in the shape tuple for input {i}.")
        
        layout_dict = {letter: number for letter, number in zip(layout_str, shape_tuple)}
        result.append(layout_dict)
    
    return result


if __name__ == "__main__":
    main()
