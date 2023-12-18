#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import torch
from numpy import ndarray
from torchvision.ops.boxes import batched_nms
import itertools


# input is size: batch_size x num_anchors_after_top_k x [x_min_abs, y_min_abs, x_max_abs, y_max_abs, top_class_index, top_class_score]
def post_process_efficientdet(
    input, iou_threshold, max_det_per_image, prediction_confidence_threshold
):
    """
    Args
        input: the decoded box proposals in Efficientdet format. 
        Dimensions are batch_size x num_anchors_after_top_k x 6; where the 6 stands for [x_min_abs, y_min_abs, x_max_abs, y_max_abs, top_class_index, top_class_score]
    Returns
        A list of length batch_size containing the final boudingboxes as Torch.Tensor [x_min_abs, y_min_abs, x_max_abs, y_max_abs, top_class_score, top_class_index]
    """
    batch_detections = []
    for i in range(input.shape[0]):
        boxes, classes, scores = input[i][:, :4], input[i][:, 4], input[i][:, 5]
        
        # NMS
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=iou_threshold)

        # Top K filtering
        top_detection_idx = top_detection_idx[:max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0

        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        # Confidence threshold based filtering
        mask = detections[:,4] > prediction_confidence_threshold
        batch_detections.append(detections[mask])
    
    return batch_detections


# input is size: batch_size x num_anchors x [x_min_norm,  y_min_norm,  x_max_norm,  y_max_norm, class_0_score, class_1_score ... class_n_score]
# class_0_score is additional background class which is later ignored
def transform_ssd_to_efficientdet(input, width, height):
    # Prepare the scaling tensor
    scale_tensor = torch.tensor([width, height, width, height], device=input.device)

    # Extract bounding boxes and apply scaling
    bboxes = input[:, :, :4]
    bboxes *= scale_tensor

    # Extract class scores and find top classes and scores
    class_scores = input[:, :, 5:]
    top_classes = torch.argmax(class_scores, dim=2)
    top_scores = torch.max(class_scores, dim=2).values

    # Stack transformed detections
    transformed_detections = torch.cat([bboxes, top_classes.unsqueeze(2), top_scores.unsqueeze(2)], dim=2)
    
    return transformed_detections


def transform_yolo_nanodet_to_efficientdet(input):
    """ Translates the decoded nanodet outputs to the format the GPP expects, via a set of vectorized transformations on GPU
    Args:
        input: batch_size x num_anchors x [x_min_abs,  y_min_abs,  x_max_abs,  y_max_abs, class_1_score, class_2_score ... class_n_score]
    Returns:

        """
    # Extract bounding boxes
    bboxes = input[:, :, :4]
    
    # Extract class scores and find top classes and scores
    class_scores = input[:, :, 4:]
    top_classes = torch.argmax(class_scores, dim=2)
    top_scores = torch.max(class_scores, dim=2).values
    
    # Stack transformed detections
    transformed_detections = torch.cat([bboxes, top_classes.unsqueeze(2).float(), top_scores.unsqueeze(2)], dim=2)
    
    return transformed_detections


def postprocess(input, output_format, input_size, max_det_per_image, prediction_confidence_threshold, iou_threshold):
    """
    Args
        input: List of outputs for a batch, where each output is
        batched and is either a sequence or an ndarray.
        E.g. print([output.shape for output in outputs]) for three
        outputs of bs 8 will yield something like
          [(8,10,10,5), (8,5), (8,)]

        **kwargs: Not accessible through leip evaluate just yet.

    Returns
        A Sequence(batch size). Each item in that sequence is a Sequence
        of LEIP representations, either Labels or BoundingBoxes.
    """
    height = input_size[0]
    width = input_size[1]
    
    # Perform necesary vectorized transformations to adapt decoded model outputs
    if output_format == "ssd":
        input = transform_ssd_to_efficientdet(input, width, height)
    elif output_format == "yolo" or output_format == "nanodet":
        input = transform_yolo_nanodet_to_efficientdet(input)

    # Do the Postprocessing: top K filtering -> NMS -> confidence based filtering
    batch_detections = post_process_efficientdet(input, iou_threshold, max_det_per_image, prediction_confidence_threshold)
    
    return batch_detections

