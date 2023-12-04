# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python


import os
import sys

import torch as T
import torchvision.transforms as transforms

from PIL import Image

from argparse import ArgumentParser
from pathlib import Path

from pylre import LatentRuntimeEngine

def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--path_to_model", type=str, default=".",
                        help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image",
        type=str,
        default="/latentai/recipes/yolov5/inference/python/bus.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="/latentai/recipes/yolov5/inference/python/labels.txt",
        help="Path to labels text file.",
    )
    args = parser.parse_args()
    
    # sys.path.append(str(Path(args.path_to_model))) # If postprocessor is in the model
    # from processors import general_detection_postprocessor
    
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir))
    sys.path.insert(0, project_dir)
    
    from utils import general_detection_postprocessor, utils
        
    # Model Factory
    model_runtime = LatentRuntimeEngine(str(Path(args.path_to_model) / "modelLibrary.so"))
    print(model_runtime.get_metadata())    

    labels = utils.load_labels(args.labels)
    
    # Load Image and Labels
    image = Image.open(args.input_image)
    orig_size = image.size
    print("orig_size")
    print(orig_size)
    
    # Pre-process
    layout_shapes = utils.get_layout_dims(model_runtime.input_layouts, model_runtime.input_shapes)
    image_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))
    print("image_size")
    print(image_size)
    
    resize_transform = transforms.Resize(image_size)
    resized_image = resize_transform(image)
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply the normalization transformation
    resized_image_normalized = normalize_transform(resized_image)
    # Run Inference
    model_runtime.infer(resized_image_normalized)
    
    # Get outputs as a list of PyDLPack
    outputs = model_runtime.get_outputs()
    output = outputs[0]
    outputdl = T.from_dlpack(output)
    
    # Post-process
    device = model_runtime.device_type
    deploy_env =  'torch' # 'torch' 'leip' 'af'   
    output = general_detection_postprocessor.postprocess(outputdl, max_det_per_image=10, prediction_confidence_threshold=0.5, iou_threshold=0.2, height=image_size[0], width=image_size[1], model_output_format="yolo", device=device, deploy_env=deploy_env)
    output_filename = utils.plot_boxes(deploy_env, image, orig_size, image_size, labels, output, args)
    print("Annotated image written to", output_filename)

if __name__ == "__main__":
    main()
