import os
from pathlib import Path


########################
#### Configurations ####
########################
class Processors:
    def __init__(self, albumentations, visualization, preprocess, postprocess):
        self.use_albumentations_library = albumentations # True: use artifact, False: use custom
        if self.use_albumentations_library:
            self.preprocess_library_torch = False # artifact uses albumentations, this configuration ignored 
        else:
            self.preprocess_library_torch = preprocess # True: pytorch, False: cv2
        if not self.preprocess_library_torch:
            self.visualization_library_cv2 = True
        else:
            self.visualization_library_cv2 = visualization # True: cv2, False: pil
        self.postprocess_library_torch = postprocess # True: pytorch, False: cv2

    def display_config(self):
        print(f"Use albumentations library: {self.use_albumentations_library}")
        print(f"Use cv2 as visualization library: {self.visualization_library_cv2}")
        print(f"Use torch as preprocess library: {self.preprocess_library_torch}")
        print(f"Use torch as postprocess library: {self.postprocess_library_torch}")

def set_processor_configs(albumentations_path):
    albumentations_path = Path(albumentations_path) / "processors" / "af_preprocessor.json"
    if os.path.exists(albumentations_path):
        config = Processors(True, True, False, True) # use artifact, cv2, none, torch
    else:
        config = Processors(False, False, True, True) # no artifact, pil, torch, torch
    return config


########################
##### Data loaders #####
########################
def load_labels(path):
    with open(path, 'r') as label_file:
        lines = label_file.readlines()
    return lines

def load_image(path, config):
    if config.visualization_library_cv2:
        import cv2
        image = cv2.imread(path, )
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv reads in BGR
        # cv2.imwrite('/workspace/example-applications/sample_images/cv2_im.jpg',image)
        return image
    else:
        from PIL import Image
        image = Image.open(path)
        return image


########################
### Metadata readers ###
########################
def get_layout_dims(layout_list, shape_list):
    if len(layout_list) != len(shape_list):
        raise ValueError("Both input lists should have the same number of elements.")
    
    result = []
    
    for i in range(len(layout_list)):
        layout_str = layout_list[i]
        shape_tuple = shape_list[i]
        
        if len(layout_str) != len(shape_tuple):
            raise ValueError(f"Length of layout string does not match the number of elements in the shape tuple for input {i}.")
        
        layout_dict = {letter: number for letter, number in zip(layout_str, shape_tuple)}
        result.append(layout_dict)
    
    return result


########################
#### Preprocessors #####
########################
def preprocess_transforms_albumentations(image, albumentations_path):
    # Albumentations Pre-process
    import albumentations as A
    albumentations_path = Path(albumentations_path) / "processors" / "af_preprocessor.json"
    loaded_transform = A.load(albumentations_path)
    loaded_transform.processors.pop("keypoints")
    loaded_transform.processors.pop("bboxes")
    # print(loaded_transform)

    transformed_image = loaded_transform(image=image)['image']
    transformed_image = transformed_image.contiguous()

    return transformed_image

def preprocess_transforms(image, input_size, config):
    if config.preprocess_library_torch:
        import torchvision.transforms as transforms

        # apply imagenet preprocess transformations
        resize_transform = transforms.Resize(input_size)
        resized_image = resize_transform(image)
        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return normalize_transform(resized_image)
    else:
        raise RuntimeError(f"Function does not exist for config.preprocess_library_torch {config.preprocess_library_torch}.")


########################
#### Postprocessors ####
########################
def postprocess_top_one(values, config):
    if config.postprocess_library_torch:
        import torch as T

        values = T.from_dlpack(values)
        values = T.nn.functional.softmax(values, dim=1)
        max_index = T.argmax(values).item()
        max_value = values[0][max_index]        
        
        top_one = (max_index, max_value )
        return top_one
    else:
        raise RuntimeError(f"Function does not exist for config.postprocess_library_torch {config.postprocess_library_torch}.")


########################
###### Visualizers #####
########################
def print_top_one(top_one, label_file_name):
    lines = load_labels(label_file_name)

    if top_one[0] >= 0 and top_one[0] < len(lines):
        label = lines[int(top_one[0])].strip()
    else:
        label = "Unknown Label"

    return label, float(top_one[1])
    # print(" ------------------------------------------------------------ ")
    # print(" Detections ")
    # print(" ------------------------------------------------------------ ")
    # print(f" The image prediction result is: id {top_one[0]}")
    # print(f" Name: {label}")
    # print(f" Score: {top_one[1]}")
    # print(" ------------------------------------------------------------ ")


########################
#####    Timer     #####
########################

import time
import math

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_times = []
        self.ms_factor = 0.000001

    def start(self):
        self.start_time = time.time()

    def stop(self):
        end_time = time.time()
        elapsed_ms = (end_time - self.start_time) * 1000  # Convert to milliseconds
        self.elapsed_times.append(elapsed_ms)

    def averageElapsedMilliseconds(self):
        if not self.elapsed_times:
            return 0.0
        else:
            return sum(self.elapsed_times[1:]) / (len(self.elapsed_times) - 1)

    def standardDeviationMilliseconds(self):
        if len(self.elapsed_times) < 2:
            return 0.0

        mean = self.averageElapsedMilliseconds()
        accum = sum((d - mean) ** 2 for d in self.elapsed_times[1:])
        return math.sqrt(accum / (len(self.elapsed_times) - 2))

# Function to round a float to a specified number of decimal places
def roundToDecimalPlaces(value, decimal_places):
    factor = 10.0 ** decimal_places
    return round(value * factor) / factor


