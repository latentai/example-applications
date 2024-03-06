# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/usr/bin/env python

# Ensure that the import statement matches the filename and class name
from utils import utils 
import json
from pylre import LatentRuntimeEngine

t_preprocessing = utils.Timer()
t_inference = utils.Timer()
t_postprocessing = utils.Timer()

def main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Run inference")
    parser.add_argument("--precision", type=str, default="float32", help="Set precision to run LRE.")
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations to average timing.",
    )

    args = parser.parse_args()

    # Load runtime
    lre = LatentRuntimeEngine(str(Path(args.model_binary_path) / "modelLibrary.so"))
    print(lre.get_metadata())

    # Set precision
    lre.set_model_precision(args.precision)
    
    # Read metadata from runtime
    layout_shapes = utils.get_layout_dims(lre.input_layouts, lre.input_shapes)
    input_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))

    config = utils.set_processor_configs(args.model_binary_path)

    # Load image
    image = utils.load_image(args.input_image_path, config)
    
    # Warm up
    # Is this needed for CPU?
    lre.warm_up(10)

    iterations = args.iterations

    for i in range(iterations):
        # Pre-process
        t_preprocessing.start()
        if config.use_albumentations_library:
            resized_image_normalized = utils.preprocess_transforms_albumentations(image, args.model_binary_path)
        else:
            resized_image_normalized = utils.preprocess_transforms(image, input_size, config)
        t_preprocessing.stop()

        # Run inference
        t_inference.start()
        lre.infer(resized_image_normalized)
        t_inference.stop()
        
        # Post-process    
        outputs = lre.get_outputs()
        output = outputs[0]
        t_postprocessing.start()
        op = utils.postprocess_top_one(output, config)
        t_postprocessing.stop()

    # Visualize
    label, score = utils.print_top_one(op, args.labels)

    # Get the average elapsed time in milliseconds
    average_preprocessing_time = t_preprocessing.averageElapsedMilliseconds()
    std_dev_preprocessing = t_preprocessing.standardDeviationMilliseconds()
    
    average_inference_time = t_inference.averageElapsedMilliseconds()
    std_dev_inference = t_inference.standardDeviationMilliseconds()
    
    average_postprocessing_time = t_postprocessing.averageElapsedMilliseconds()
    std_dev_postprocessing = t_postprocessing.standardDeviationMilliseconds()

    average_time = average_preprocessing_time + average_inference_time + average_postprocessing_time

    # Print the result
    data = {
        "UID": lre.model_id,
        "Precision": lre.model_precision,
        "Device": lre.device_type,
        "Input Image Size": image.shape,
        "Model Input Shapes": lre.input_shapes,
        "Model Input Layouts": lre.input_layouts,
        "Average Preprocessing Time ms": {
            "Mean": utils.roundToDecimalPlaces(average_preprocessing_time, 3),
            "std_dev": utils.roundToDecimalPlaces(std_dev_preprocessing, 3)
        },
        "Average Inference Time ms": {
            "Mean": utils.roundToDecimalPlaces(average_inference_time, 3),
            "std_dev": utils.roundToDecimalPlaces(std_dev_inference, 3)
        },
        "Average Total Postprocessing Time ms": {
            "Mean": utils.roundToDecimalPlaces(average_postprocessing_time, 3),
            "std_dev": utils.roundToDecimalPlaces(std_dev_postprocessing, 3)
        },
        "Total Time ms": utils.roundToDecimalPlaces(average_time, 3),
        "Class": label,
        "Score": score
    }
    json_text = json.dumps(data, indent=4)
    print(json_text)

if __name__ == "__main__":
    main()
