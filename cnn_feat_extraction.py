#!/usr/bin/env python3

import os
import sys
import threading
import numpy as np
import yaml
import cv2
import pickle
import pdb
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models

# resnet = models.resnet34(pretrained=True)
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1] # delete the last fc layer.
resnet = nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False


def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # TODO
    video_cnn = None
    for image in get_keyframes(downsampled_video_filename, keyframe_interval):
        tensor = torch.tensor(np.moveaxis(image, -1, 0)).type('torch.FloatTensor')
        tensor = tensor.unsqueeze(0)
        cnn = resnet(tensor)
        cnn = cnn.squeeze()
        if video_cnn is None:
            video_cnn = cnn.unsqueeze(0)
        else:
            video_cnn = torch.cat((video_cnn, cnn.unsqueeze(0)), dim=0)

    if video_cnn is not None:
        np.savetxt(cnn_feat_video_filename + '.csv', video_cnn.numpy())
        print (video_cnn.size())

    print (downsampled_video_filename)
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
        img = cv2.resize(img,(224,224))
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
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create CNN object

    # Check if folder for CNN features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get CNN features for all videos but only from keyframes

    print ('directory created!')

    all_video_names = ['all_test_fake.lst', 'all_trn.lst', 'all_val.lst']

    for i in range(3):
        print (i)
        fread = open(all_video_names[i], "r")
        for line in fread.readlines():
            video_name = line.split()[0]
            downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.mp4.ds.mp4')
            cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')
            if os.path.isfile(cnn_feat_video_filename + '.csv'):
                print ('existed!')
                continue

            # Get cnn features for one video
            get_cnn_features_from_video(downsampled_video_filename,
                                         cnn_feat_video_filename, keyframe_interval)
