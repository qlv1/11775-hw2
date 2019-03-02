#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # TODO

    key_frame = 0
    video_des = None
    for image in get_keyframes(downsampled_video_filename, keyframe_interval):
        key_frame += 1
        surf = cv2.SURF(10000)
        kp, des = surf.detectAndCompute(image,None)
        if des is not None:
            if video_des is None:
                video_des = des
            else:
                video_des = np.concatenate((video_des, des))

    if video_des is not None:
        np.savetxt(surf_feat_video_filename + '.csv', video_des)


    print (downsampled_video_filename)
    print (np.shape(video_des))
    pass


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()

        if ret is False:
            break
        # if frame % keyframe_interval == 0:
        yield img

    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    print ('directory created!')

    all_video_names = ['all_test_fake.lst', 'all_trn.lst', 'all_val.lst']

    for i in range(3):
        print (i)
        fread = open(all_video_names[i], "r")
        for line in fread.readlines():
            video_name = line.split()[0]
            downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.mp4.ds.mp4')
            surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')
            if os.path.isfile(surf_feat_video_filename + '.csv'):
                print ('existed!')
                continue

            # Get SURF features for one video
            get_surf_features_from_video(downsampled_video_filename,
                                         surf_feat_video_filename, keyframe_interval)
