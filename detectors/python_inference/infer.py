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
import os
import sys
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
        "--iterations",
        type=int,
        default=10,
        help="Iterations to average timing.",
    )
    parser.add_argument(
        "--maximum_detections",
        type=int,
        default=10,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.2,
        help="IOU threshold.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        help="Model model_family to use for preprocessing.",
    )
    
    args = parser.parse_args()
    
    from utils import detector_preprocessor, detector_postprocessor, utils
        
    # Load runtime
    lre = LatentRuntimeEngine(str(Path(args.model_binary_path) / "modelLibrary.so"))
    print(lre.get_metadata())

    # Set precision
    lre.set_model_precision(args.precision)

    # Read metadata from runtime
    layout_shapes = utils.get_layout_dims(lre.input_layouts, lre.input_shapes)
    input_size = (layout_shapes[0].get('H'), layout_shapes[0].get('W'))
    
    config = utils.set_processor_configs(args.model_binary_path)

    # Load Image and Labels
    image = utils.load_image(args.input_image_path, config)
    labels = utils.load_labels(args.labels)
    
    # Warm up
    # Is this needed for CPU?
    lre.warm_up(10)
    
    iterations = args.iterations

    for i in range(iterations):
        # Pre-process
        t_preprocessing.start()
        if config.use_albumentations_library:
            sized_image, transformed_image = detector_preprocessor.preprocess_transforms_albumentations(image, args.model_binary_path)
        else:
            if args.model_family:
                sized_image, transformed_image = detector_preprocessor.preprocess_transforms(image, args.model_family, input_size, config)
            else:
                raise RuntimeError(f"--model_family argument is not provided to preprocess without albumentations.")
        t_preprocessing.stop()

        # Run inference
        t_inference.start()
        lre.infer(transformed_image)
        t_inference.stop()

        # Get outputs as a list of PyDLPack
        outputs = lre.get_outputs()
        output = outputs[0]

        # Post-process  
        t_postprocessing.start()
        postprocessor_path = Path(args.model_binary_path) / "processors" / "general_detection_postprocessor.py"
        if os.path.exists(postprocessor_path):
            postprocessor_path = postprocessor_path.resolve()
            postprocessor_path = str(postprocessor_path.parents[0])
            sys.path.append(postprocessor_path)
            from general_detection_postprocessor import post_process
            import torch as T
            output_torch = T.from_dlpack(output)
            output = post_process(output_torch, max_det_per_image=args.maximum_detections, prediction_confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold)
        else:
            output = detector_postprocessor.postprocess(output, max_det_per_image=args.maximum_detections, prediction_confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, config=config)
            t_postprocessing.stop()
    
    # Visualize
    output_image = utils.plot_boxes(sized_image, output, labels)
    output_filename = utils.save_image(output_image, args.input_image_path, config)
    
    # Get the average elapsed time in milliseconds
    average_preprocessing_time = t_preprocessing.averageElapsedMilliseconds()
    std_dev_preprocessing = t_preprocessing.standardDeviationMilliseconds()
    
    average_inference_time = t_inference.averageElapsedMilliseconds()
    std_dev_inference = t_inference.standardDeviationMilliseconds()
    
    average_postprocessing_time = t_postprocessing.averageElapsedMilliseconds()
    std_dev_postprocessing = t_postprocessing.standardDeviationMilliseconds()

    average_time = average_preprocessing_time + average_inference_time + average_postprocessing_time

    # Create a dictionary representing the model details
    j = {
        "UUID": lre.model_id,
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
        "Annotated Image": output_filename
    }

    json_str = json.dumps(j, indent=2)
    print(json_str)


if __name__ == "__main__":
    main()
