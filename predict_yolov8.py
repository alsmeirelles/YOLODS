from ultralytics import YOLO as yolo
from torchvision import ops
import torch
import argparse
import os
import numpy as np

def box_check(gtr,pbox,oshape):
    """
    GTR is XYWHN, read from annotation file;
    PBOX should be xyxy from model prediction;
    OSHAPE is the original image´s shape
    """
    #Round box XY coordinates
    pbox = np.round(pbox)
    #Convert GTR to xyxy
    gtr = [l.strip().split(' ') for l in gtr]
    gtr = np.array(gtr).astype(np.float32)
    convcoord = np.ones((len(gtr),4),dtype=np.uint)
def predict(config):

    timages = os.listdir(os.path.join(config.ts,"images"))
    tset = {ti:os.path.join(config.ts,"labels","{}.txt".format(ti[:-4])) for ti in timages}

    if config.verbose > 0:
        print("Loading model: {}".format(config.yolo))

    model = yolo(config.yolo)

    results = model.predict(source=config.ts)

    if config.verbose > 0:
        print("Calculating metrics...")

    for r in results:
        im_name = os.path.basename(r.path)
        with open(tset[im_name],"r") as fd:
            labels = fd.readlines()
        if box_check(labels,r.boxes.xyxy) > config.iou:
            pass


if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Train an yolov8n model with given \
        dataset.')
    parser.add_argument('-iou', dest='iou', type=float,
        help='Minimum IoU to consider a match (Default: 0.45)', default=0.45,required=False)
    parser.add_argument('-ts', dest='ts', type=str, default='',
        help='Path to a test set.',required=True)
    parser.add_argument('-yolo', dest='yolo', type=str, default='yolov8n.pt',
        help='Yolo model to use.',required=True)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-bc', dest='bc', type=int,
        help='Class index to consider positive (Default: 1)', default=1,required=False)

    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.ts):
        print("Test dir not found: {}".format(config.ts))
        sys.exit(0)
    elif not (os.path.isdir(os.path.join(config.ts,"images")) and os.path.isdir(os.path.join(config.ts,"labels"))):
        print("Test dir does not have the required structure.")
        sys.exit(0)

    predict(config)
