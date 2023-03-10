# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

import torch
from typing import Optional

from representations.boundingboxes.boundingbox import BoundingBox
from representations.boundingboxes.utils import BBFormat
from effdet.anchors import generate_detections
from effdet.anchors import Anchors
from preprocessors import factory as preprocessor_factory

    
def postprocess(
        outputs,
        sample_paths,
        **kwargs):
    """
    Args
        outputs: List of outputs for a batch, where each output is batched and is either a sequence or an ndarray.
                 E.g. print([output.shape for output in outputs]) for three outputs of bs 8 will yield something like
                      [(8,10,10,5), (8,5), (8,)]

        sample_paths: List of paths to reference items in the given batch.
                 E.g. For the above example ["path1", "path2", "path3", "path4", "path5", "path6", "path7", "path8""]

        **kwargs: Not accessible through leip evaluate just yet.

    Returns
        A Sequence(batch size). Each item in that sequence is a Sequence
        of LEIP representations, either Labels or BoundingBoxes.
    """

    # Set values for configurable parameters # TODO: Ask VP can these be part of signature kwargs?
    prediction_confidence_threshold = 0.2
    nms_method = False
    width = 512
    height = 512
    num_levels = 5
    num_classes = 90
    max_detection_points = 5000
    max_det_per_image = 15

    postprocessed_batch = []

    for sample, path in zip([outputs], sample_paths):

        scores = sample[0:5]
        boxes = sample[5:10]

        batch_size = torch.tensor(scores[0]).shape[0]

        cls_outputs_all = torch.cat([
            torch.tensor(scores[level]).permute(0, 2, 3, 1).reshape([batch_size, -1, num_classes])
            for level in range(num_levels)], 1)

        box_outputs_all = torch.cat([
            torch.tensor(boxes[level]).permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
            for level in range(num_levels)], 1)

        _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
        # FIXME change someday, will have to live with annoying warnings for a while as testing impl breaks torchscript
        indices_all = torch.div(cls_topk_indices_all, num_classes, rounding_mode='trunc')
        # indices_all = cls_topk_indices_all // num_classes
        classes_all = cls_topk_indices_all % num_classes

        box_outputs_all_after_topk = torch.gather(
            box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))
        # cls_outputs_alternative = torch.gather(
        #    cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 20))

        cls_outputs_all_after_topk = torch.gather(
            cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
        cls_outputs_all_after_topk = torch.gather(
            cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))
        # picked_box_probs = []
        # picked_labels = []

        anchor_params = {"min_level": 3, "max_level": 7, "num_scales": 3, "anchor_scale": 4}
        anchors = Anchors(aspect_ratios=[(1, 1), (1.4, 0.7), (0.7, 1.4)], image_size=(width, height), **anchor_params)

        detections = batch_detection(
            batch_size, cls_outputs_all_after_topk, box_outputs_all_after_topk, anchors.boxes, indices_all, classes_all,
            # target['img_scale'], target['img_size'],
            max_det_per_image=max_det_per_image, soft_nms=nms_method)

        batched_filtered_detections = []
        for sample in detections:
            sample_filtered_detections = sample[sample[:, 4] > prediction_confidence_threshold]
            batched_filtered_detections.append(sample_filtered_detections)

        bbs = []
        for k in range(batched_filtered_detections[0].shape[0]):
            bbs.append(BoundingBox(
                class_id=batched_filtered_detections[0][k][5],
                coords=batched_filtered_detections[0][k][:4],
                image_name=path,
                image_size=(width, height),
                confidence=batched_filtered_detections[0][k][4],
                bb_format=BBFormat.pascal_voc))

        postprocessed_batch.append(bbs)
    return postprocessed_batch


# Apply NMS for batch detection
def batch_detection(
        batch_size: int,
        class_out,
        box_out,
        anchor_boxes,
        indices,
        classes,
        img_scale: Optional[torch.Tensor] = None,
        img_size: Optional[torch.Tensor] = None,
        max_det_per_image: int = 100,
        soft_nms: bool = False):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(
            class_out[i], box_out[i], anchor_boxes, indices[i], classes[i],
            img_scale_i, img_size_i, max_det_per_image=max_det_per_image, soft_nms=soft_nms)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)
