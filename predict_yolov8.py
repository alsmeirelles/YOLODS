from ultralytics import YOLO as yolo
from torchvision import ops
from sklearn import metrics
import torch
import argparse
import os
import sys
import numpy as np

def box_check(gtr,pbox,classes,oshape,verbose=0):
    """
    GTR is XYWHN, read from annotation file;
    PBOX should be xyxy from model prediction;
    OSHAPE is the original image´s shape (tupple)
    """
    #Round box XY coordinates
    pbox = np.round(pbox)

    if verbose:
        print(f"Normalized GTR:\n {gtr}")
        print(f"Original shape: {oshape}")
        print(f"PBOX (shape {pbox.shape}):\n {pbox}")

    convcoord = np.zeros((len(gtr),5),dtype=np.uint)
    convcoord[:,1:] = convcoord[:,1:] + oshape*2 #Duplicates the tupple, it´s not a multiplication x2
    convcoord[:,3:] = convcoord[:,3:]/2
    gtr = np.round((gtr*convcoord))

    gtr[:,1:3] = gtr[:,1:3] - gtr[:,3:]
    gtr[:,3:] *= 2 #Restablish Width x Height values
    if verbose:
        print(f"GTR (XYWH):\n {gtr}")

    gtr[:,3:] += gtr[:,1:3]
    if verbose:
        print(f"GTR (XYXY):\n {gtr}")

    iou = None
    gtr = torch.tensor(gtr,dtype=torch.float)
    if len(pbox) != len(gtr):
        iou = ops.generalized_box_iou(gtr[:,1:],pbox)
    else:
        iou = ops.box_iou(gtr[:,1:],pbox)

    if verbose:
        print(gtr[:,:1].T,classes)
        print("IoU: {}".format(iou))

    return iou

def predict(config):

    if os.path.isdir(config.ts):
        timages = os.listdir(os.path.join(config.ts,"images"))
        paths = [os.path.join(config.ts,"images",im) for im in timages]
        tset = {ti:os.path.join(config.ts,"labels","{}.txt".format(ti[:-4])) for ti in timages}
    elif os.path.isfile(config.ts):
        paths = [config.ts]
        spt = os.path.split(config.ts)
        tset = {spt[1]:os.path.join(os.path.dirname(spt[0]),"labels","{}.txt".format(spt[1][:-4]))}

    if config.verbose:
        print("Loading model: {}".format(config.yolo))

    model = yolo(config.yolo)

    results = model.predict(source=paths)

    if config.verbose:
        print("Calculating metrics...")

    y = []
    pred = []
    for r in results:
        im_name = os.path.basename(r.path)
        with open(tset[im_name],"r") as fd:
            labels = fd.readlines()
        if not labels:
            continue
            
        #Generate an ndarray from labels
        gtr = [l.strip().split(' ') for l in labels]
        gtr = np.array([gtr]).astype(np.float32)
        y.extend(gtr[:,:1].T[0])
        pred.extend(r.boxes.conf)

        if config.verbose:
            print(f"fpr: {fpr}; tpr: {tpr}; thresh: {thresh}")

        box_check(gtr,r.boxes.xyxy,r.boxes.cls,r.orig_shape,verbose=config.verbose)

    fpr,tpr,thresh = metrics.roc_curve(y,pred,pos_label=0)

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Train an yolov8n model with given \
        dataset.')
    parser.add_argument('-iou', dest='iou', type=float,
        help='Minimum IoU to consider a match (Default: 0.45)', default=0.45,required=False)
    parser.add_argument('-ts', dest='ts', type=str, default='',
        help='Path to a test set.',required=False)
    parser.add_argument('-yolo', dest='yolo', type=str, default='yolov8n.pt',
        help='Yolo model to use.',required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-bc', dest='bc', type=int,
        help='Class index to consider positive (Default: 0)', default=0,required=False)
    parser.add_argument('-debug', dest='debug', action='store_true',
        help='Run debugging procedures', default=False,required=False)

    config, unparsed = parser.parse_known_args()

    if config.debug:
        config.ts = "C:/Users/Meirelles/Documents/GT-IA/Yolo/datasets/Merged/test/images/2-352-_jpg.rf.31f549fd47a4ec7c84b8cf6fe417cb57.jpg"
        config.verbose = 1
        iou = predict(config)
        print(f"IoU: {iou}")
        sys.exit(0)
    elif not config.ts or not os.path.isdir(config.ts):
        print("Test dir not found: {}".format(config.ts))
        sys.exit(0)
    elif not (os.path.isdir(os.path.join(config.ts,"images")) and os.path.isdir(os.path.join(config.ts,"labels"))):
        print("Test dir does not have the required structure.")
        sys.exit(0)

    predict(config)
