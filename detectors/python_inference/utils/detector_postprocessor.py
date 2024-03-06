#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.


def post_process_torch(
    input, iou_threshold, max_det_per_image, prediction_confidence_threshold
):
    import torch
    from torchvision.ops.boxes import batched_nms

    batch_detections = []
    # Why the loop?
    for i in range(input.shape[0]):
        boxes, classes, scores = input[i][:, :4], input[i][:, 4], input[i][:, 5]

        # Filter based on confidence threshold
        filtered_scores = scores > prediction_confidence_threshold
        boxes = boxes[filtered_scores]
        scores = scores[filtered_scores]
        classes = classes[filtered_scores]
        # test for zero filtered values
        
        # NMS
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=iou_threshold)

        # Filter based on NMS & Top K
        top_detection_idx = top_detection_idx[:max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0

        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        # # Filter based on confidence threshold
        # mask = detections[:,4] > prediction_confidence_threshold
        batch_detections.append(detections)
    
    return batch_detections


def take_max_scores_torch(input):
    import torch

    # Extract bounding boxes
    bboxes = input[:, :, :4]
    
    # Extract class scores and find top classes and scores
    class_scores = input[:, :, 4:]
    top_classes = torch.argmax(class_scores, dim=2)
    top_scores = torch.max(class_scores, dim=2).values
    
    # Stack transformed detections
    transformed_detections = torch.cat([bboxes, top_classes.unsqueeze(2).float(), top_scores.unsqueeze(2)], dim=2)
    
    return transformed_detections


def postprocess(output, max_det_per_image, prediction_confidence_threshold, iou_threshold, config):
    if config.postprocess_library_torch:
        import torch as T
        output_torch = T.from_dlpack(output) 
        output_torch = take_max_scores_torch(output_torch)

        # Do postprocessing: confidence based filtering -> NMS -> top K filtering
        batch_detections = post_process_torch(output_torch, iou_threshold, max_det_per_image, prediction_confidence_threshold)
        return batch_detections
    else:
        raise RuntimeError(f"Function does not exist for config.postprocess_library_torch {config.postprocess_library_torch}.")
    
