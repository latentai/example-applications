# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

import os
import cv2
import sys
import datetime
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pylre import LatentRuntimeEngine
import torch
import torchvision.transforms as transforms
from PIL import Image


def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--path_to_model", type=str, default=".",
                        help="Path to LRE object directory.")
    parser.add_argument(
        "--input_image",
        type=str,
        help="Path to input image.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to labels text file.",
    )

    args = parser.parse_args()
    model_path = str(Path(args.path_to_model))
    sys.path.append(model_path)

    from processors import general_detection_postprocessor

    # from preprocessors.factory import get_preprocessor
    # from postprocessors.factory import get_postprocessor
    # from representations.boundingboxes.utils import BBFormat
    # import albumentations as A

    # pad_transform = A.Compose([
    #     A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    #     A.augmentations.geometric.transforms.PadIfNeeded(
    #         min_height=512, min_width=512, value=[124, 116, 104], border_mode=0)
    # ])
    
    # Load your image
    pil_image = Image.open(args.input_image)

    # Define target size and pad value
    target_height = 512
    target_width = 512
    pad_value = 124  # RGB values for padding

    # Apply custom padding
    padded_image_pil = custom_pad(pil_image, target_height, target_width, pad_value)
    transform = transforms.ToTensor()
    padded_image_tensor = transform(padded_image_pil)

    # image = Image.open(args.input_image)
    model = LatentRuntimeEngine(str(Path(args.path_to_model) / "modelLibrary.so"))
    print(model.get_metadata())
    # image = cv2.imread(args.input_image)  # Replace with the path to your image
    # transformed_image = pad_transform(image, max_size=500)
    
    
    labels = load_labels(args.labels)

    # Model inference
    outputs = model.infer(padded_image_tensor)
    
    # Apply post-processing to model outputs # to be DONE
    # predictions = model._postprocess(outputs, args.input_image)

    # rgb_img = Image.fromarray(pad_transform_img).convert("RGB")
    # out_im = np.array(cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2RGB))
    # threshold = 0.3

    # for bb in predictions:
    #     for i in range(0, len(bb)):
    #         if bb[i].get_confidence() > threshold:
    #             out_im = plot_one_box(
    #                 bb[i].get_coordinates(
    #                     BBFormat.absolute_xyx2y2, image_size=rgb_img.size),
    #                 out_im,
    #                 color=(255, 0, 0),
    #                 label=labels[int(bb[i].get_class_id())],
    #             )
    #         i = i+1

    # p = os.path.splitext(args.input_image)
    # output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"
    # cv2.imwrite(output_filename, out_im)
    # print("Annotated image written to", output_filename)

def custom_pad(image, target_height, target_width, pad_value):
    # Calculate the padding values
    pad_height = max(0, target_height - image.size[1])
    pad_width = max(0, target_width - image.size[0])

    # Calculate the resizing dimensions
    resize_height = target_height - pad_height
    resize_width = target_width - pad_width

    # Resize the image to fit within the target dimensions
    resized_image = image.resize((resize_width, resize_height))

    # Create a new blank image with the desired size and fill with pad_value
    padded_image = Image.new("RGB", (target_width, target_height), (pad_value, pad_value, pad_value))

    # Paste the resized image into the center of the padded image
    padded_image.paste(resized_image, ((pad_width // 2), (pad_height // 2)))

    return padded_image

def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")


def plot_one_box(box, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

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


if __name__ == "__main__":
    main()
