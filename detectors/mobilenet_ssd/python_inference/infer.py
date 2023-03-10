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

def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--lre_object", type=str, default=".", help="Path to LRE object directory.")
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
    sys.path.append(str(Path(args.lre_object) / "latentai.lre"))
    
    import latentai_runtime
    import numpy as np
    from PIL import Image
    from preprocessors.factory import get_preprocessor
    from postprocessors.factory import get_postprocessor
    from representations.boundingboxes.boundingbox import BoundingBox
    from representations.boundingboxes.utils import BBFormat
    
    model = latentai_runtime.Model()

    labels = load_labels(args.labels)

    image = Image.open(args.input_image)    
    data = model._preprocess([image])
    
    inference = model.infer(data)
    
    output = model._postprocess(inference, args.input_image)

    rgb_img = image.convert("RGB")
    out_im = np.array(cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2RGB))
    threshold = 0.3
    for bb in output:
        for i in range(0,len(bb)):
            if bb[i].get_confidence() > threshold:
                out_im = plot_one_box(
                    bb[i].get_coordinates(BBFormat.absolute_xyx2y2, image_size=rgb_img.size),
                    out_im,
                    color=(255, 0, 0),
                    label=labels[bb[i].get_class_id()],
                )
    
    p = os.path.splitext(args.input_image)
    output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"
    cv2.imwrite(output_filename, out_im)
    print("Annotated image written to", output_filename)


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


if __name__ == "__main__":
    main()
