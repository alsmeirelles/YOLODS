#!/usr/bin/env python3
"""
Script to run data augmentation and also divide generated images
into YOLO Dataset format.

Yolo annotation format is: class_id X Y W H
Coordinates are normalized by corresponding image dimension
"""

import os
import sys
import numpy as np
import cv2
import argparse
import albumentations as A
import random

from matplotlib import pyplot as plt

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def bbox_convert(gtr,oshape,verbose=0):
    """
    Returns a tuple of (F1,F2)
    GTR is XYWHN, read from annotation file;
    F1: XYXY equivalnet to model prediction;
    F2: XYWH compatible with CV2 and Albumentations
    OSHAPE is the original image´s shape (tupple)
    """
    #Round box XY coordinates
    pbox = np.round(pbox)

    if verbose:
        print(f"Normalized GTR:\n {gtr}")
        print(f"Original shape: {oshape}")

    convcoord = np.zeros((len(gtr),5),dtype=np.uint)
    convcoord[:,1:] = convcoord[:,1:] + oshape*2 #Duplicates the tupple, it´s not a multiplication x2
    convcoord[:,3:] = convcoord[:,3:]/2
    gtr = np.round((gtr*convcoord))

    gtr[:,1:3] = gtr[:,1:3] - gtr[:,3:]
    gtr[:,3:] *= 2 #Restablish Width x Height values
    f2 = gtr[:,1:]
    if verbose:
        print(f"GTR (XYWH):\n {f2}")

    gtr[:,3:] += gtr[:,1:3]
    f1 = gtr[:,1:]
    if verbose:
        print(f"GTR (XYXY):\n {f1}")

    return (f1,f2)

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """
    Visualizes a single bounding box on the image.

    bbox format: XYWH
    """
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def read_bbox(lpath):
    """
    Read label file and return bounding boxes and corresponding classes
    """
    with open(lpath,"r") as fd:
        labels = fd.readlines()

    gtr = [l.strip().split(' ') for l in labels]
    gtr = np.array([gtr]).astype(np.float32)
    classes = gtr[:,:1].T[0]

    bboxes = gtr[:,1:]

    return (bboxes,classes)

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

def apply_transform(data,transform,impath,lbpath,debug=False):

    for i in data:
        im = cv2.imread(os.path.join(impath,img_name))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        bbox,category_ids = read_bbox(os.path.join(lbpath,"{}.txt".format(i[-4:])))
        tf = transform(image=im,bboxes=bbox,category_ids=category_ids)

def run_augmentation(config):
    """
    Runs the augmentations
    """
    imgs = list(filter(lambda d: (d.endswith("jpg") or d.endswith("png")),os.path.listdir(config.img)))
    category_ids = config.classes.keys()
    category_id_to_name = config.classes

    if config.debug:
        imgs = random.choices(imgs,k=10)

    transform = A.compose(
        [A.ColorJitter(),
        ], #list of transformations
        bbox_params = A.BboxParams(format="yolo",label_fields=['category_ids'],min_visibility=0.2,min_area=100))
    if config.cpu > 1:
        from Multiprocess import multiprocess_run
        r = multiprocess_run(apply_transform,(transform,config.img,config.ann,config.debug),imgs,config.cpu,pbar=True,step_size=20,txt_label="Generating images")
    else:
        r = apply_transform(imgs,transform,config.img,config.ann)

if __name__ == "__main__":

    def_classes = {0:"Gun",1:"Knife"}
    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract frames to be used to train an YOLO model .')

    parser.add_argument('-img', dest='img', type=str, default='',
        help='Location of original images.',required=True)
    parser.add_argument('-ann', dest='ann', type=str, default='',
        help='Location of annotation files.',required=True)
    parser.add_argument('-d', dest='dest', type=str, default='DS',
        help='Generate YOLO DS in this directory.',required=False)
    parser.add_argument('-cpu', dest='cpu', type=int, default=1,
        help='Run augmentations in parallel. Number of workers (default=1).',required=False)
    parser.add_argument('-classes', dest='classes', type=json.loads, default=def_classes,
        help='Class names and IDs (should be a dictionary formatted as string).',required=False)
    parser.add_argument('-db', dest='debug', action='store_true',
        help='Run debugging procedures. Apply augmentation to a random set of 10 images.', default=False,required=False)

    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.img):
        print("Directory not found: {}".format(config.img))
        sys.exit()
    if not os.path.isdir(config.ann):
        print("Directory not found: {}".format(config.ann))
        sys.exit()

    run_augmentation(config)
