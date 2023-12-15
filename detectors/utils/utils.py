import cv2
import datetime
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from pathlib import Path

VISUALIZATION_LIBRARY = 'pil' # 'pil' 'cv2'

########################
##### Data loaders #####
########################
def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")

def load_image_pil(path):
    # perhaps add a cv version too?
    from PIL import Image
    image = Image.open(path)
    return image

def load_image_cv(path):
    image = cv2.imread(path, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv reads in BGR
    # cv2.imwrite('/workspace/example-applications/sample_images/cv2_im.jpg',image)
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
##### Box plotters #####
########################
def plot_one_box(box, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # list of COLORS
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return img

def plot_boxes(deploy_env, image, labels, output, args):
    if deploy_env == 'leip':
        from representations.boundingboxes.utils import BBFormat
        rgb_img = image.convert("RGB")
        out_im = np.array(cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2RGB))
        threshold = 0.3
        for bb in output:
            for i in range(0,len(bb)):
                if bb[i].get_confidence() > threshold:
                    out_im = plot_one_box(
                        bb[i].get_coordinates(BBFormat.absolute_xyx2y2, image_size=rgb_img.size),
                        out_im,
                        color=(255, 0, 0),
                        label=labels[bb[i].get_class_id()],
                    )

    elif deploy_env == 'torch' and VISUALIZATION_LIBRARY == 'pil':
        from torchvision.utils import draw_bounding_boxes
        pil_transform = transforms.PILToTensor()
        out_im = pil_transform(image)
        for bb in output:
            for i in range(0,len(bb)):
                box = bb[i][0:4]
                box[0] = box[0] #*orig_size[0]/image_size[0]
                box[1] = box[1] #*orig_size[1]/image_size[1]
                box[2] = box[2] #*orig_size[0]/image_size[0]
                box[3] = box[3] #*orig_size[1]/image_size[1]
                box = box.unsqueeze(0)
                label = [labels[int(bb[i][5])]]
                out_im = draw_bounding_boxes(out_im, box, label, 
                    width=5, colors="blue", fill=False) 
                    # font="serif", font_size=30)
        pil_to_transform = transforms.ToPILImage()
        out_im = pil_to_transform(out_im)
    
    elif deploy_env == 'torch' and VISUALIZATION_LIBRARY == 'cv2':
        # rgb_img = image.convert("RGB")
        out_im = np.array(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
        for bb in output:
            for i in range(0,len(bb)):
                box = bb[i][0:4]
                # box = box.unsqueeze(0)
                label = labels[int(bb[i][5])]
                out_im = plot_one_box(
                    box,
                    out_im,
                    color=(255, 0, 0),
                    label=label,
                )
    
    p = os.path.splitext(args.input_image_path)
    output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"
    if deploy_env == 'leip':
        cv2.imwrite(output_filename, out_im)
    elif deploy_env == 'torch' and VISUALIZATION_LIBRARY == 'pil':
        out_im.save(output_filename)
    elif deploy_env == 'torch' and VISUALIZATION_LIBRARY == 'cv2':
        cv2.imwrite(output_filename, out_im)
    return output_filename