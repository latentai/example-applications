#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

TRANSFORM_LIBRARY = 'torch' # cv2, torch, or albumentations

def preprocess(image, input_transform, input_size):
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
    
    if TRANSFORM_LIBRARY == 'albumentations':
        sized_image, transformed_image = albumentations_preprocess_transforms(image, input_transform, input_size)
    elif TRANSFORM_LIBRARY == 'torch':
        sized_image, transformed_image = torch_preprocess_transforms(image, input_transform, input_size)
    else: # default is cv2
        sized_image, transformed_image = opencv_preprocess_transforms(image, input_transform, input_size)

    return sized_image, transformed_image

## Albumentations processors
def load_albumentations_preprocess(image, albumentations_path):
    # Albumentations Pre-process
    import albumentations as A
    loaded_transform = A.load(albumentations_path)
    loaded_transform.processors.pop("keypoints")
    loaded_transform.processors.pop("bboxes")
    print(loaded_transform)

    transformed_image = loaded_transform(image=image)['image']
    transformed_image = transformed_image.contiguous()

    return transformed_image

def albumentations_preprocess_transforms(image, input_transform, input_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ## transforms
    resize_transform = A.Compose([
        A.Resize(p=1, height=input_size[0], width=input_size[0], interpolation=1),
    ], p=1.0)
    resize_center_color1_transform = A.Compose([
        A.LongestMaxSize(p=1, max_size=max(input_size), interpolation=1),
        A.PadIfNeeded(p=1, min_height=input_size[0], min_width=input_size[1], border_mode=0, value=[124, 116, 104]),
    ], p=1.0)
    resize_center_color2_transform = A.Compose([
        A.LongestMaxSize(p=1, max_size=max(input_size), interpolation=1),
        A.PadIfNeeded(p=1, min_height=input_size[0], min_width=input_size[1], border_mode=0, value=[128, 128, 128]),
    ], p=1.0)
    float_normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(p=1, transpose_mask=False),
    ], p=1.0)
    int_normalize = A.Compose([
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=1),
        ToTensorV2(p=1, transpose_mask=False),
    ], p=1.0)
    # this transform is incorrect

    ## image
    if input_transform == 'ssd':   
        sized_image = resize_transform(image=image)   
    elif input_transform == 'yolo':   
        sized_image = resize_center_color2_transform(image=image)
    elif input_transform == 'nanodet':   
        sized_image = resize_transform(image=image)
    else: # default is for efficient det
        sized_image = resize_center_color1_transform(image=image)
    
    sized_image = sized_image['image']

    ## models
    if input_transform == 'ssd':    
        transformed_image = resize_transform(image=image)
        transformed_image = float_normalize(image=transformed_image['image'])
    elif input_transform == 'yolo':
        transformed_image = resize_center_color2_transform(image=image)
        transformed_image = int_normalize(image=transformed_image['image'])
    elif input_transform == 'nanodet':
        transformed_image = resize_transform(image=image)
        transformed_image = float_normalize(image=transformed_image['image'])
    else: # default is for efficient det
        transformed_image = resize_center_color1_transform(image=image)
        transformed_image = float_normalize(image=transformed_image['image'])

    transformed_image = transformed_image['image']
    transformed_image = transformed_image.contiguous()
    
    return sized_image, transformed_image

## Pytorch processors
def torch_preprocess_transforms(image, input_transform, input_size):
    import torchvision.transforms as transforms

    ## transforms
    resize_transform = transforms.Resize(input_size)
    resize_center_color1_transform = transforms.Compose([
        transforms.Resize(input_size)
        # transforms.Resize(384),  # Equivalent to LongestMaxSize
        # transforms.Pad ((0, 0, 512-384, 0), fill=(124, 116, 104), padding_mode='constant'),  # Equivalent to PadIfNeeded
    ])
    ## Resize is like shortestmaxsize, so I'm forcing 1080 to be 512 by resizing with 512/1080*810.
    ## Since height then needs padding, I do 512-384. This does not center things.
    ## Still the boxes are off.
    resize_center_color2_transform = transforms.Compose([
        transforms.Resize(input_size)
        # transforms.Resize(480),  # Equivalent to LongestMaxSize
        # transforms.Pad((80, 0, 80, 0), fill=(128, 128, 128), padding_mode='constant'),  # Equivalent to PadIfNeeded
    ])
    ## padding makes it look wrong. Why?
    # if input is padded, the visual image should also be padded
    float_normalize = transforms.Compose([
        transforms.ToTensor(),  # Equivalent to ToTensorV2
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    int_normalize = transforms.Compose([
        transforms.ToTensor(),  # Equivalent to ToTensorV2
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ])

    ## image
    if input_transform == 'ssd':   
        sized_image = resize_transform(image)   
    elif input_transform == 'yolo':   
        sized_image = resize_center_color2_transform(image)
    elif input_transform == 'nanodet':   
        sized_image = resize_transform(image)
    else: # default is for efficient det
        sized_image = resize_center_color1_transform(image)

    ## models
    if input_transform == 'ssd':    
        transformed_image = resize_transform(image)
        transformed_image = float_normalize(transformed_image)
    elif input_transform == 'yolo':
        transformed_image = resize_center_color2_transform(image)
        transformed_image = int_normalize(transformed_image)
    elif input_transform == 'nanodet':
        transformed_image = resize_transform(image)
        transformed_image = float_normalize(transformed_image)
    else: # default is for efficient det
        transformed_image = resize_center_color1_transform(image)
        transformed_image = float_normalize(transformed_image)
    
    return sized_image, transformed_image

## OpenCV processors
def opencv_preprocess_transforms(image, input_transform, input_size):
    import cv2
    
    ## transforms
    def resize_transform(image):
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        return image

    def resize_center_color1_transform(image):
        min_height, min_width = input_size[0], input_size[1]
        pad_color = [124, 116, 104]
        height, width, _ = image.shape
        top_pad = max(0, min_height - height)
        bottom_pad = max(0, min_height - height - top_pad)
        left_pad = max(0, min_width - width)
        right_pad = max(0, min_width - width - left_pad)
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=pad_color)
        return image

    def resize_center_color2_transform(image):
        return image

    def float_normalize(image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image / 255.0 - mean) / std
        return image

    def int_normalize(image):
        mean = [0, 0, 0]
        std = [1, 1, 1]
        image = (image / 255.0 - mean) / std
        return image

    ## image
    if input_transform == 'ssd':   
        sized_image = resize_transform(image)   
    elif input_transform == 'yolo':   
        sized_image = resize_center_color2_transform(image)
    elif input_transform == 'nanodet':   
        sized_image = resize_transform(image)
    else: # default is for efficient det
        sized_image = resize_center_color1_transform(image)

    ## models
    if input_transform == 'ssd':    
        transformed_image = resize_transform(image)
        transformed_image = float_normalize(transformed_image)
    elif input_transform == 'yolo':
        transformed_image = resize_center_color2_transform(image)
        transformed_image = int_normalize(transformed_image)
    elif input_transform == 'nanodet':
        transformed_image = resize_transform(image)
        transformed_image = float_normalize(transformed_image)
    else: # default is for efficient det
        transformed_image = resize_center_color1_transform(image)
        transformed_image = float_normalize(transformed_image)

    transformed_image = cv2.dnn.blobFromImage(transformed_image, 
                                  scalefactor=1/255,
                                  size=input_size,
                                  swapRB=True)

    return sized_image, transformed_image