#!/bin/python
# Randomly select

import numpy
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of video names"
        print "select_ratio -- the ratio of frames to be randomly selected from each audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    for line in fread.readlines():
        file_path = "cnn/" + line.split()[0] + ".cnn.csv"
        print (file_path)

        if os.path.exists(file_path) == False:
            continue

        array = numpy.genfromtxt(file_path, delimiter=" ")
        if array.ndim <= 1:
            continue

        numpy.random.shuffle(array)
        select_size = min(array.shape[0], 60)
        feat_dim = array.shape[1]

        for n in xrange(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')

    fwrite.close()
