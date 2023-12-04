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
def post_process_efficientdet_format_for_leip(
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



def transform_into_leip_representation(batch_detections, coco80_to_coco91_map, height, width):
    from representations.boundingboxes.boundingbox import BoundingBox
    from representations.boundingboxes.utils import BBFormat
    postprocessed_batch = []
    for sample_detections in batch_detections:
        bbs = []
        for bbox in sample_detections:
            class_id = coco80_to_coco91_map[bbox[5].int() - 1] if coco80_to_coco91_map else bbox[5].int()
            bbs.append(
                BoundingBox(
                    class_id=class_id,
                    coords=bbox[:4],
                    image_size=(width, height),
                    confidence=bbox[4],
                    bb_format=BBFormat.pascal_voc,
                )
            )
        postprocessed_batch.append(bbs)
    return postprocessed_batch


def transform_into_AF_representation(batch_detections):
    from af.core.utils.annotations import BoundingBox2d as bbox2d
    postprocessed_batch = []
    for sample_detections in batch_detections:
        sample_detections = sample_detections.cpu()
        bbs = []
        for bbox in sample_detections:
            bbs.append(bbox2d.from_format('pascal_voc', bbox[:4], label=int(bbox[5]), confidence=bbox[4]))
        postprocessed_batch.append(bbs)
    return postprocessed_batch


# input is size: batch_size x num_anchors x [x_min_norm,  y_min_norm,  x_max_norm,  y_max_norm, class_0_score, class_1_score ... class_n_score]
# class_0_score is additional background class which is later ignored
def transform_ssd_to_efficientdet(input, width, height, **kwargs):
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


def postprocess(input, **kwargs):
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
    max_det_per_image = int(kwargs["max_det_per_image"])
    prediction_confidence_threshold = float(kwargs["prediction_confidence_threshold"])
    iou_threshold = float(kwargs["iou_threshold"])
    convert_coco80_to_coco91 = kwargs.get("convert_coco80_to_coco91", False) # for efficientdet or any other models not pretrained on coco-detection this should be False (efficientdet was pretrained on coco-detection-90)
    coco80_to_coco91_map = [x for x in range(1, 91) if x not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]] if convert_coco80_to_coco91 else None
    height = int(kwargs["height"])
    width = int(kwargs["width"])
    output_format = kwargs["model_output_format"]
    deploy_env = kwargs["deploy_env"]
    # Deal with different input format when model is ingested via SDK
    if isinstance(input, list) and isinstance(input[0], ndarray):
        assert len(input) == 1, f"there is more than one model output, but the general postprocessor you are using expects a single model output"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("Moving model outputs to CUDA for postprocessing because it is available" if torch.cuda.is_available() else "postprocessing on CPU")
        input = torch.from_numpy(input[0]).to(device) # only the first element of the list is needed, it also requires numpy -> torch.tensor conversion
        deploy_env = 'leip'
    
    # Perform necesary vectorized transformations to adapt decoded model outputs
    if output_format == "ssd":
        input = transform_ssd_to_efficientdet(input, width, height)
    elif output_format == "yolo" or output_format == "nanodet":
        input = transform_yolo_nanodet_to_efficientdet(input)

    # Do the Postprocessing: top K filtering -> NMS -> confidence based filtering
    batch_detections = post_process_efficientdet_format_for_leip(input, iou_threshold, max_det_per_image, prediction_confidence_threshold)
    
    if deploy_env == 'leip':
        postprocessed_batch = transform_into_leip_representation(batch_detections, coco80_to_coco91_map, height, width)
    elif deploy_env == 'torch':
        postprocessed_batch = batch_detections
    else:
        postprocessed_batch = transform_into_AF_representation(batch_detections)
    return postprocessed_batch

