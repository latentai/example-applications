# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python
import numpy as np
import torch.nn.functional as F
import torch as T
import torchvision.transforms as transforms
from PIL import Image
import math

def main():
    import sys
    from argparse import ArgumentParser
    from pathlib import Path

    from pylre import LatentRuntimeEngine

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=str, default=".", help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image",
        type=str,
        default="/latentai/recipes/classifiers/images/penguin.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="/latentai/recipes/classifiers/inference/python/labels.txt",
        help="Path to labels text file.",
    )
    args = parser.parse_args()

    # Model Factory
    m = LatentRuntimeEngine(str(Path(args.model) / "modelLibrary.so"))
    print(m.get_metadata())

    # WarmUp Phase
    m.warm_up(1)
    
    # Load the image using PIL (Python Imaging Library)
    input_image_path = args.input_image
    image = Image.open(input_image_path)
    image_size = (224, 224)
    resize_transform = transforms.Resize(image_size)
    resized_image = resize_transform(image)
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply the normalization transformation
    resized_image_normalized = normalize_transform(resized_image)

    # Run inference
    m.infer(resized_image_normalized)
    
    # Post-process    
    outputs = m.get_outputs()

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

if __name__ == "__main__":
    main()
