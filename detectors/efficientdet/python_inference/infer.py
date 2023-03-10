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


def main():

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--lre_object", type=str, default=".",
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
    sys.path.append(str(Path(args.lre_object) / "latentai.lre"))

    import latentai_runtime
    from PIL import Image
    from preprocessors.factory import get_preprocessor
    from postprocessors.factory import get_postprocessor
    from representations.boundingboxes.utils import BBFormat
    import albumentations as A

    pad_transform = A.Compose([
        A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
        A.augmentations.geometric.transforms.PadIfNeeded(
            min_height=512, min_width=512, value=[124, 116, 104], border_mode=0)
    ])

    image = Image.open(args.input_image)
    model = latentai_runtime.Model()
    labels = load_labels(args.labels)

    # Apply preprocessing to the image
    data = model._preprocess([image])
    pad_transform_img = pad_transform(image=np.array(image))["image"]
    # Model inference
    outputs = model.infer(data)
    # Apply post-processing to model outputs
    predictions = model._postprocess(outputs, args.input_image)

    rgb_img = Image.fromarray(pad_transform_img).convert("RGB")
    out_im = np.array(cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2RGB))
    threshold = 0.3

    for bb in predictions:
        for i in range(0, len(bb)):
            if bb[i].get_confidence() > threshold:
                out_im = plot_one_box(
                    bb[i].get_coordinates(
                        BBFormat.absolute_xyx2y2, image_size=rgb_img.size),
                    out_im,
                    color=(255, 0, 0),
                    label=labels[int(bb[i].get_class_id())],
                )
            i = i+1

    p = os.path.splitext(args.input_image)
    output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"
    cv2.imwrite(output_filename, out_im)
    print("Annotated image written to", output_filename)


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
