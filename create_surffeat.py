#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    if os.path.exists('surf_vector/') == False:
        os.mkdir('surf_vector/')

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    # all_video_names = ['all_test_fake.lst', 'all_trn.lst', 'all_val.lst']
    all_video_names = ['all_val.lst']

    for i in range(len(all_video_names)):
        print (i)
        fread = open(all_video_names[i], "r")
        i = 0

        for line in fread.readlines():
            mfcc_path = "surf/" + line.split()[0] + ".surf.csv"
            print (mfcc_path)
            if os.path.exists(mfcc_path) == False:
                continue
            array = np.genfromtxt(mfcc_path, delimiter=" ")
            if array.ndim == 1:
                array = np.expand_dims(array, axis=0)
            if array.shape[1] == 0:
                continue
            Y = kmeans.predict(array)
            Y_count = np.bincount(Y, minlength = cluster_num)
            Y_norm = Y_count.astype(float) / len(array)
            if i == 0:
                Y_all = Y_norm
                i = 1
            else:
                Y_all = np.vstack((Y_all, Y_norm))
            np.savetxt("surf_vector/" + line.split()[0] + ".surf.csv", Y_norm.transpose())

    print (Y_all)
    print (np.sum(Y_all, axis = 1))

    print "K-means features generated successfully!"
