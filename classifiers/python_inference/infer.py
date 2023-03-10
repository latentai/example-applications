# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python

def main():
    import sys
    from argparse import ArgumentParser
    from pathlib import Path

    from PIL import Image

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--lre_object", type=str, default=".", help="Path to LRE object directory.")
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

    sys.path.append(str(Path(args.lre_object) / "latentai.lre"))
    import latentai_runtime

    m = latentai_runtime.Model()

    image = Image.open(args.input_image)
    labels = load_labels(args.labels)

    prediction = m.predict([image])[0][0]

    print("Predicted =>", labels[prediction.class_id], prediction.confidence)


def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")


if __name__ == "__main__":
    main()
