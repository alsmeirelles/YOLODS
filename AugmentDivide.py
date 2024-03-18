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
import json
import uuid

from matplotlib import pyplot as plt
from typing import Tuple

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])

    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new_shape)) / min(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
    delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def bbox_convert(gtr,oshape,verbose=0):
    """
    Returns a tuple of (F1,F2)
    GTR is XYWHN, read from annotation file;
    F1: XYXY equivalnet to model prediction;
    F2: XYWH compatible with CV2 and Albumentations
    OSHAPE is the original image´s shape (tupple) (usually returned by .shape attribute)
    """
    #Invert shape to WH
    oshape = oshape[::-1]
    if verbose:
        print(f"Normalized GTR:\n {gtr}")
        print(f"Original shape (inverted): {oshape}")

    convcoord = np.zeros((len(gtr),4),dtype=np.uint)
    convcoord[:,:] = convcoord[:,:] + oshape*2 #Duplicates the tupple, it´s not a multiplication x2
    convcoord[:,2:] = convcoord[:,2:]/2
    gtr = np.round((gtr*convcoord))

    gtr[:,:2] = gtr[:,:2] - gtr[:,2:]
    gtr[:,2:] *= 2 #Restablish Width x Height values
    f2 = gtr[:,:].copy()
    if verbose:
        print(f"GTR (XYWH):\n {f2}")

    gtr[:,2:] += gtr[:,:2]
    f1 = gtr[:,:]
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
    gtr = np.array(gtr).astype(np.float32)
    classes = gtr[:,:1].T[0]

    bboxes = gtr[:,1:]
    return (bboxes,classes)

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        xyxy,xywh = bbox_convert([bbox],img.shape[:2],verbose=True)
        img = visualize_bbox(img, xywh[0], class_name) #one box at a time
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def apply_transform(data,transform,dest,impath,lbpath,imsize,classes,debug=False):

    res = []
    for i in data:
        im = cv2.imread(os.path.join(impath,i))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        bbox,category_ids = read_bbox(os.path.join(lbpath,"{}.txt".format(i[:-4])))

        if debug:
            print("Previewing before resize....")
            visualize(im,bbox,category_ids,config.classes)

        tf = transform(image=im,bboxes=bbox,category_ids=category_ids)
        res.append(tf)

        if debug:
            visualize(tf['image'],tf['bboxes'],category_ids,config.classes)
        else:
            #Write generated images to disk
            fname = "{}-{}".format(i[:-4],uuid.uuid5(uuid.NAMESPACE_DNS,"GTIA-IC"))
            with open(os.path.join(dest,f"{fname}.txt"),"w") as fd:
                for bb,cid in zip(tf['bboxes'],tf['category_ids']):
                    fd.readline(f"{cid} {bb}")


    return res

def _send_images_labels(imgs,ddir):
    pass

def run_augmentation(config):
    """
    Runs the augmentations
    """
    imgs = list(filter(lambda d: (d.endswith("jpg") or d.endswith("png")),os.listdir(config.img)))
    category_ids = config.classes.keys()
    category_id_to_name = config.classes

    if config.debug:
        imgs = random.choices(imgs,k=config.dbs)

    #Transformtions to perform
    transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.25,contrast_limit=0.25),
        A.ToGray(p=0.3),
        A.InvertImg(p=0.3),
        A.Sequential([A.LongestMaxSize(max_size=max(config.tdim),interpolation=1,always_apply=True),
            A.PadIfNeeded(min_height=config.tdim[1],min_width=config.tdim[0],border_mode=0,value=(0,0,0),always_apply=True),
            A.RandomSizedBBoxSafeCrop(height=config.tdim[1],width=config.tdim[0],p=0.5)],p=1.0),
        A.OneOrOther(first=A.OneOrOther(first=A.HorizontalFlip(p=0.3),second=A.VerticalFlip(p=0.3)),
            second=A.Rotate(limit=(30,60),border_mode=cv2.BORDER_CONSTANT,value=(0,0,0),p=0.3))], #list of transformations
        bbox_params = A.BboxParams(format="yolo",label_fields=['category_ids'],min_visibility=0.2,min_area=50),
        p=1.0)

    if config.cpu > 1:
        from Multiprocess import multiprocess_run
        r = multiprocess_run(apply_transform,(transform,config.dest,config.img,config.ann,config.tdim,config.classes,config.debug),
            imgs,config.cpu,pbar=True,step_size=20,txt_label="Generating images")[0]
    else:
        r = apply_transform(imgs,transform,config.dest,config.img,config.ann,config.tdim,config.classes,debug=config.debug)

    #Distribute original and generated images to dataset
    if not config.debug:
        total = len(imgs) + len(r)
        op = len(imgs)/total

        #Test set
        tssize = round(config.ptest*total)
        from_imgs = round(op*tssize)
        from_augm = tssize - from_imgs

    return r

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
    parser.add_argument('-dbs', dest='dbs', type=int, default=10,
        help='Sample this many images for debugging (default=10).',required=False)
    parser.add_argument('-tdim', dest='tdim', nargs='+', type=int,
        help='Default width and heigth, all images will be resized to this.',
        default=(640,640), metavar=('Width', 'Height'))
    parser.add_argument('-ptest', dest='ptest', type=float, default=0.2,
        help='Percentage of images to be part of the test set (default=20%).',required=False)
    parser.add_argument('-val', dest='val', type=int, default=200,
        help='Percentage of images to be part of the validation set (default=200).',required=False)

    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.img):
        print("Directory not found: {}".format(config.img))
        sys.exit()
    if not os.path.isdir(config.ann):
        print("Directory not found: {}".format(config.ann))
        sys.exit()
    if not os.path.isdir(config.dest):
        os.mkdir(config.dest)

    res = run_augmentation(config)
