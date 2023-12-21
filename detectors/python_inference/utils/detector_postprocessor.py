#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import torch
from torchvision.ops.boxes import batched_nms


def post_process_efficientdet(
    input, iou_threshold, max_det_per_image, prediction_confidence_threshold
):
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


def transform_ssd_to_efficientdet(input, width, height):
    # Prepare the scaling tensor
    scale_tensor = torch.tensor([width, height, width, height], device=input.device)

    # Extract bounding boxes and apply scaling
    bboxes = input[:, :, :4]
    bboxes *= scale_tensor

    # Extract class scores and find top classes and scores
    # class_0_score is additional background class which is ignored
    class_scores = input[:, :, 5:]
    top_classes = torch.argmax(class_scores, dim=2)
    top_scores = torch.max(class_scores, dim=2).values

    # Stack transformed detections
    transformed_detections = torch.cat([bboxes, top_classes.unsqueeze(2), top_scores.unsqueeze(2)], dim=2)
    
    return transformed_detections


def transform_yolo_nanodet_to_efficientdet(input):
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

