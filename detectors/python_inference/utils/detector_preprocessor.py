#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import utils 
from pathlib import Path

def preprocess_transforms_albumentations(image, albumentations_path):
    # Albumentations Pre-process
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    albumentations_path = Path(albumentations_path) / "processors" / "af_preprocessor.json"
    loaded_transform = A.load(albumentations_path)
    # print(loaded_transform)
    loaded_transform.processors.pop("keypoints")
    loaded_transform.processors.pop("bboxes")

    transformed_image = loaded_transform(image=image)['image']
    transformed_image = transformed_image.contiguous()

    loaded_transform.transforms = [t for t in loaded_transform.transforms if not isinstance(t, A.Normalize)]
    loaded_transform.transforms = [t for t in loaded_transform.transforms if not isinstance(t, ToTensorV2)]
    sized_image = loaded_transform(image=image)['image']

    return sized_image, transformed_image

## Pytorch transforms
def preprocess_transforms(image, input_transform, input_size, config):
    if config.preprocess_library_torch:
        import torchvision.transforms as transforms

        ## Transforms to resize and pad
        resize_transform = transforms.Resize(input_size)
        image_size = image.size
        width_height_image = image_size[0]/image_size[1]
        width_height_expected = input_size[0]/input_size[1]
        if width_height_image > width_height_expected:
            new_image_size = (input_size[0], input_size[1] / width_height_image)
            resize_var = int(input_size[1] / width_height_image)
        else:
            new_image_size = (input_size[0] * width_height_image, input_size[1])
            resize_var = int(input_size[0] * width_height_image)
        filler = tuple(int(a - b) for a, b in zip(input_size, new_image_size))
        resize_center_color1_transform = transforms.Compose([
            transforms.Resize(resize_var),  # Equivalent to LongestMaxSize
            transforms.Pad ((0, 0, filler[0], filler[1]), fill=(124, 116, 104), padding_mode='constant'),  # Equivalent to PadIfNeeded
        ])
        resize_center_color2_transform = transforms.Compose([
            transforms.Resize(resize_var),  # Equivalent to LongestMaxSize
            transforms.Pad((int(filler[0]/2), int(filler[1]/2), int(filler[0]/2), int(filler[1]/2)), fill=(128, 128, 128), padding_mode='constant'),  # Equivalent to PadIfNeeded
        ])
        
        ## Transforms to normalize and tensorize.
        imagenet_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_normalize = transforms.Compose([
            transforms.ToTensor(),
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
            transformed_image = imagenet_normalize(transformed_image)
        elif input_transform == 'yolo':
            transformed_image = resize_center_color2_transform(image)
            transformed_image = pixel_normalize(transformed_image)
        elif input_transform == 'nanodet':
            transformed_image = resize_transform(image)
            transformed_image = imagenet_normalize(transformed_image)
        else: # default is for efficient det
            transformed_image = resize_center_color1_transform(image)
            transformed_image = imagenet_normalize(transformed_image)
        
        return sized_image, transformed_image
    else:
        raise RuntimeError(f"Function does not exist for config.preprocess_library_torch {config.preprocess_library_torch}.")
