#!/usr/bin/env python3
#-*- coding: utf-8
#ALSMEIRELLES

import os
import sys
import numpy as np
import cv2
import argparse

#Yolo
from ultralytics import YOLO as yolo

def run_frame_extraction(config):
    """
    Searches for video files in data directory.
    For each video found, inspect every X video frames in search for cars
    If a car is found, extract the next Y frames to destination directory
    """
    extracted_frames = 0
    analyzed_videos = 0

    #Searches for known video files in source directory
    extensions = ['mp4','avi']

    #Check data source
    if config.data is None:
        print("No data source defined. Define the -data option.")
        sys.exit(1)

    #Check what is the last image # in destination dir
    dst_imgs = os.listdir(config.dst_dir)
    dst_imgs = list(filter(lambda x: x.endswith('.jpg'),dst_imgs))
    if len(dst_imgs) > 0:
        dst_imgs = list(map(lambda x: int(x.split('.')[0]),dst_imgs))
        dst_imgs.sort()
        last_img = dst_imgs[-1]
    else:
        last_img = 0

    if os.path.isfile(config.mclasses):
        classes_path = config.mclasses
    else:
        classes_path = os.path.join('data','classes',config.classes)

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = {class_names[c].strip():c for c in range(len(class_names))}

    if config.find_class in class_names:
        class_n = class_names[config.find_class]
    else:
        print("Specified class is not detectable by YOLO: {}".format(config.find_class))
        return None

    pred_args = {
        "model_path": os.path.join(config.model_path,'YOLO-trained_weights_final.h5'),
        "anchors_path": config.anchors,
        "classes_path": classes_path,
        "gpu_num" : config.gpu_count}

    predictor = YOLO(**pred_args)

    input_files = os.listdir(config.data)
    for f in input_files:
        if config.info:
            print("Extracting from {}...".format(f))
        if os.path.isfile(os.path.join(config.data,f)) and f.split('.')[1] in extensions:
            new_imgs = _run_extractor(predictor,os.path.join(config.data,f),config.dst_dir,
                                          config.frame_inter,config.frame_count,last_img,class_n,
                                          config.threshold)
            extracted_frames += new_imgs
            last_img += new_imgs
            analyzed_videos += 1

    print("Extracted {} frames from {} videos".format(extracted_frames,analyzed_videos))
    print("Total available images: {}".format(last_img))


def _run_extractor(predictor,video_path,dst_dir,frame_interval,frame_count,last_img,class_n,ts):
    """
    Runs extraction
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Video could not be opened: {}".format(video_path))
        return 0

    vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fcount = 0
    extracted = 0
    while fcount < vid_frames:
        vid.set(cv2.CAP_PROP_POS_FRAMES, fcount)
        ret, frame = vid.read()
        if ret and predictor.detect_class(frame,class_n,ts):
            cv2.imwrite(os.path.join(dst_dir,'{}.jpg'.format(last_img+extracted)),frame)
            fcount += 1
            extracted += 1
            for j in range(frame_count):
                if j >= vid_frames:
                    break
                vid.set(cv2.CAP_PROP_POS_FRAMES, fcount)
                ret, frame = vid.read()
                if ret:
                    cv2.imwrite(os.path.join(dst_dir,'{}.jpg'.format(last_img+extracted)),frame)
                    extracted += 1
                else:
                    print("Error reading frame {} from video {}".format(fcount,video_path))
                fcount += 1
            fcount += frame_interval
        else:
            fcount += frame_interval
            print("Steping to frame {}".format(fcount))

    vid.release()
    return extracted

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract frames to be used to train an YOLO model .')

    parser.add_argument('-vdata', dest='vdata', type=str, default='',
        help='Location of video files.',required=True)
    parser.add_argument('-fdata', dest='fdata', type=str, default='',
        help='Location of annotation files.',required=True)


    config, unparsed = parser.parse_known_args()

    run_frame_extraction(config)
