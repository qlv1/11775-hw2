#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video
i = 0

for file in ['P001', 'P002', 'P003']:
    pred = np.loadtxt('surf_pred/' + file + '_test_surf.lst')
    if i == 0:
        pred_all = np.zeros((len(pred), 6))
        pred_all[:, 0] = pred
        i += 1
    else:
        pred_all[:, i] = pred
        i += 1

    pred = np.loadtxt('cnn_pred/' + file + '_test_cnn.lst')
    pred_all[:, i] = pred
    i += 1

pred_final = np.zeros(len(pred)).astype(int)

for i in range(len(pred)):
    pred_final[i] = np.argmax(pred_all[i]) + 2
    pred_final[i] = pred_final[i] / 2
    # else:
    #     pred_final[i] = 0

print (pred_final)
np.savetxt('pred_6_new.csv', pred_final)
