#!/usr/bin/env python3
#-*- coding: utf-8
#ALSMEIRELLES
import concurrent.futures
import os
import sys
import cv2
import argparse
import random


# noinspection PyTypeChecker
def run_frame_extraction(config: argparse.Namespace):
    """
    Searches for video files in data directory.
    For each video found, inspect every config.ns video frames every second
    @type config: argparse namespace object
    """
    extracted_frames = 0
    analyzed_videos = 0

    #Searches for known video files in source directory
    extensions = ['mp4','avi']

    input_files = None
    #Check data source
    if os.path.isfile(config.fdata):
        input_files = [config.fdata]
    elif os.path.isdir(config.vdata):
        input_files = list(filter(lambda x: x.split('.')[1] in extensions, os.listdir(config.vdata)))
    else:
        print("Video data not found.")
        sys.exit(1)

    if config.cpu > 1 and len(input_files) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        bar = tqdm(desc="Processing video files...", total=len(input_files), position=0)
        with ThreadPoolExecutor(max_workers=config.cpu) as executor:
            videos = {executor.submit(_run_extractor,
                                      os.path.join(config.vdata, f),
                                      config.out,
                                      config.ns,
                                      config.verbose,
                                      ): f for f in input_files}
        for task in concurrent.futures.as_completed(videos):
            bar.update(1)
            extracted_frames += task.result()
            analyzed_videos += 1
    else:
        for f in input_files:
            if config.verbose:
                print(f"Extracting from {f}...")

            new_imgs = _run_extractor(os.path.join(config.vdata, f), config.out, config.ns,config.verbose)
            extracted_frames += new_imgs
            analyzed_videos += 1

    print("Extracted {} frames from {} videos".format(extracted_frames, analyzed_videos))

def _run_extractor(video_path:str, dst_dir:str, nfps:int, verbosity:int = 0):
    """
    Runs extraction
    """
    random.seed()
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Video could not be opened: {}".format(video_path))
        return 0

    video_name = os.path.basename(video_path)[:-4]
    vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    if verbosity >= 1:
        print(f"Total frames in video: {vid_frames}; {fps} FPS")

    fcount = 0
    extracted = 0
    while fcount < vid_frames:
        fi = random.sample(range(fcount, min(fcount+fps, vid_frames)), k=nfps)

        for j in fi:
            vid.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret, frame = vid.read()
            if ret:
                cv2.imwrite(os.path.join(dst_dir, '{}-{}.jpg'.format(video_name, j)), frame)
                extracted += 1
            else:
                print("Error reading frame {} from video {}".format(j, video_path))
        fcount += fps

    vid.release()
    return extracted

if __name__ == "__main__":

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract frames from video files.')

    parser.add_argument('-vdata', dest='vdata', type=str, default='',
        help='Path to folder containing video files.',required=False)
    parser.add_argument('-fdata', dest='fdata', type=str, default='',
        help='Path to a specific video file.',required=False)
    parser.add_argument('-out', dest='out', type=str, default='',
        help='Save frames here.',required=True)
    parser.add_argument('-ns',dest='ns', type=int, default=3,
        help='Extract this many frames per second (Set to zero for all).')
    parser.add_argument('-cpu', dest='cpu', type=int, default=1,
        help='Multiprocess extraction. Define the number of processes (Default=1).')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    if not config.vdata and not config.fdata:
        print('You should define a path to a dir (vdata) or file (fdata)')
        sys.exit(1)

    run_frame_extraction(config)
