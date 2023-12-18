#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

def preprocess(image, input_transform, input_size):
    """
    Args
        image: PIL image.
        input_transform: By default, efficientdet preprocess transforms are applied.
        Any other recipe requires setting appropriate input_transform and
        any new model requires writing custom transforms to match af_preprocess.json.
        input_size: Input (height, width) expected by the model.
    Returns
        sized_image: Input PIL image with resizing and padding for visualization.
        transformed_image: Torch tensor to feed into the model.
    """
    
    # Redirect to a relevant transform library.
    sized_image, transformed_image = torch_preprocess_transforms(image, input_transform, input_size)

    return sized_image, transformed_image

## Pytorch transforms
def torch_preprocess_transforms(image, input_transform, input_size):
    import torchvision.transforms as transforms

    ## Transforms to resize and pad
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
    
    ## Transforms to normalize and tensorize.
    float_normalize = transforms.Compose([
        transforms.ToTensor(),  # Equivalent to ToTensorV2
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    int_normalize = transforms.Compose([
        transforms.ToTensor(),  # Equivalent to ToTensorV2
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ])

    ## Transforms to an image for visualization.
    if input_transform == 'ssd':   
        sized_image = resize_transform(image)   
    elif input_transform == 'yolo':   
        sized_image = resize_center_color2_transform(image)
    elif input_transform == 'nanodet':   
        sized_image = resize_transform(image)
    else: # default is for efficient det
        sized_image = resize_center_color1_transform(image)

    ## Transforms to an image for inference.
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
