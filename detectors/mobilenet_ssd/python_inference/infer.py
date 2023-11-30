# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python

import cv2
import os
import sys
import datetime
from argparse import ArgumentParser
from pathlib import Path
from pylre import LatentRuntimeEngine
import torchvision.transforms as transforms
import torch as T

def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--path_to_model", type=str, default=".", help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image",
        type=str,
        default="/latentai/mscoco/val2017/oneimg/bus.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="/latentai/recipes/ssd/inference/python/labels.txt",
        help="Path to labels text file.",
    )
    
    args = parser.parse_args()
    sys.path.append(str(Path(args.path_to_model)))

    from processors import general_detection_postprocessor
    
    import numpy as np
    from PIL import Image

    
    model_runtime = LatentRuntimeEngine(str(Path(args.path_to_model) / "modelLibrary.so"))
    print(model_runtime.get_metadata())
    
    labels = load_labels(args.labels)

    image = Image.open(args.input_image)    

    layout_shapes = get_layout_dims(model_runtime.input_layouts, model_runtime.input_shapes)
    image_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))
    print(image_size)
    
    resize_transform = transforms.Resize(image_size)
    resized_image = resize_transform(image)
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply the normalization transformation
    resized_image_normalized = normalize_transform(resized_image)
    
    # Run inference
    model_runtime.infer(resized_image_normalized)
    
    # Get outputs as a list of PyDLPack
    outputs = model_runtime.get_outputs()
    output = outputs[0]
    outputdl = T.from_dlpack(output)
    
    # import pdb; pdb.set_trace()
    device = model_runtime.device_type    
    output = general_detection_postprocessor.postprocess(outputdl, max_det_per_image=10, prediction_confidence_threshold=0.5, iou_threshold=0.2, height=image_size[0], width=image_size[1], model_output_format="ssd", device=device)
    print(output)


    # output = model._postprocess(inference, args.input_image)

    # rgb_img = image.convert("RGB")
    # out_im = np.array(cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2RGB))
    # threshold = 0.3
    # for bb in output:
    #     for i in range(0,len(bb)):
    #         if bb[i].get_confidence() > threshold:
    #             out_im = plot_one_box(
    #                 bb[i].get_coordinates(BBFormat.absolute_xyx2y2, image_size=rgb_img.size),
    #                 out_im,
    #                 color=(255, 0, 0),
    #                 label=labels[bb[i].get_class_id()],
    #             )
    
    # p = os.path.splitext(args.input_image)
    # output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"
    # cv2.imwrite(output_filename, out_im)
    # print("Annotated image written to", output_filename)


def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")


def plot_one_box(box, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # list of COLORS
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return img

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
