#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC, OneClassSVM
from sklearn.metrics.pairwise import chi2_kernel
import cPickle
import sys
import random

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    fread = open('all_trn.lst',"r")

    i = 0
    for line in fread.readlines():
        mfcc_name, label = line.split(' ')
        if feat_dir == 'surf_vector/':
            mfcc_path = feat_dir + mfcc_name.replace('\n','') + ".surf.csv"
        else:
            mfcc_path = feat_dir + mfcc_name.replace('\n','') + ".cnn.csv"

        if os.path.exists(mfcc_path) == False:
            continue
        else:
            X = np.genfromtxt(mfcc_path)
            Y_label = label.replace('\n','')
            if Y_label != 'NULL' or random.random() > 0:
                if Y_label == event_name:
                    Y = 1
                else:
                    Y = 0

                if i == 0:
                    X_all = X
                    Y_all = Y
                    i = 1
                else:
                    X_all = np.vstack((X_all, X))
                    Y_all = np.append(Y_all, Y)

    # print(X_all, Y_all)

    clf = SVC(kernel = chi2_kernel)
    # clf = SVC()
    clf.fit(X_all, Y_all)

    print (clf.score(X_all, Y_all))
    print (clf.predict(X_all))

    fread.close()

    cPickle.dump(clf, open(output_file, "wb"))

    print 'SVM trained successfully for event %s!' % (event_name)
