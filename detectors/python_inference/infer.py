# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python

import os
import sys
import cv2

from argparse import ArgumentParser
from pathlib import Path

from pylre import LatentRuntimeEngine

USE_ALBUMENTATIONS = False # True is not supported
if USE_ALBUMENTATIONS:
    PREPROCESS_TORCH = False # albumentations cannot use torch
else:
    PREPROCESS_TORCH = True # False -> Use CV # False is not fully supported
POSTPROCESS_TORCH = True # False is not supported

def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--model_binary_path", type=str, default=".", help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="../../sample_images/bus.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Path to labels text file.",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        default="efficientdet",
        help="Model model_format to use for pre/post processing.",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="torch",
        help="Representation format to use for pre/post processing.",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=10,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.2,
        help="IOU threshold.",
    )
    
    args = parser.parse_args()
    # sys.path.append(str(Path(args.model_binary_path))) # If postprocessor is in the model
    # from processors import general_detection_postprocessor
    
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    sys.path.append(project_dir)
    
    from utils import detector_preprocessor, detector_postprocessor, utils
        
    # Model Factory
    lre = LatentRuntimeEngine(str(Path(args.model_binary_path) / "modelLibrary.so"))
    use_fp16 = bool(int(os.getenv("TVM_TENSORRT_USE_FP16", 0)))

    if(use_fp16):
        lre.set_model_precision("float16")
    print(lre.get_metadata())

    layout_shapes = utils.get_layout_dims(lre.input_layouts, lre.input_shapes)
    input_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))
    print("expected input size: " + str(input_size))

    device = lre.device_type
    print("expected device: " + str(device))
    
    # Load Image and Labels
    if PREPROCESS_TORCH:
        image = utils.load_image_pil(args.input_image_path)
        print("image size: " + str(image.size))
    else:
        image = utils.load_image_cv(args.input_image_path)
        print("image size: " + str(image.shape))
    labels = utils.load_labels(args.labels)
    if USE_ALBUMENTATIONS:
        albumentations = Path(args.model_binary_path) / "processors" / "af_preprocessor.json"

    # Pre-process
    if USE_ALBUMENTATIONS:
        # sized_image = no clear way to generate this
        transformed_image = detector_preprocessor.load_albumentations_preprocess(image, albumentations)
        # image should be cv, albumentations should return a torch
    else:
        sized_image, transformed_image = detector_preprocessor.preprocess(image, args.model_format, input_size)
    print("input size: " + str(transformed_image.shape))

    # Run Inference
    lre.infer(transformed_image)
    
    # Get outputs as a list of PyDLPack
    outputs = lre.get_outputs()
    output = outputs[0]

    # Post-process
    if POSTPROCESS_TORCH:
        import torch as T
        output_torch = T.from_dlpack(output)   
        output = detector_postprocessor.postprocess(output_torch, max_det_per_image=args.max_det, prediction_confidence_threshold=args.confidence, iou_threshold=args.iou, height=input_size[0], width=input_size[1], output_format=args.model_format, device=device, deploy_env=args.representation)
    else:
        import numpy as np
        output_numpy = np.from_dlpack(output)
        ## cv post processing is not provided yet
    
    # Generate visualizations
    output_filename = utils.plot_boxes(args.representation, sized_image, labels, output, args)
    print("Annotated image written to", output_filename)

if __name__ == "__main__":
    main()