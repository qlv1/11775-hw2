#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal.
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.

# Paths to different tools;
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH

echo "#####################################"
echo "#       MED with surf Features      #"
echo "#####################################"
mkdir -p surf_pred
# iterate over the events
feat_dim_surf=1000
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_val_svm.py $event "surf_vector/" $feat_dim_surf surf_pred/svm.$event.test.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python test_svm_submit.py surf_pred/svm.$event.test.model "surf_vector/" $feat_dim_surf surf_pred/${event}_test_surf.lst || exit 1;
  # python test_svm_submit.py surf_pred/svm.$event.model "surf_vector/" $feat_dim_surf surf_pred/${event}_test_surf.lst || exit 1;
  # compute the average precision by calling the mAP package
  # ap ${event}_val_label surf_pred/${event}_val_surf.lst
done

echo ""
echo "#####################################"
echo "#       MED with cnn Features       #"
echo "#####################################"
mkdir -p cnn_pred
# iterate over the events
feat_dim_cnn=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_val_svm.py $event "cnn_vector/" $feat_dim_cnn cnn_pred/svm.$event.test.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python test_svm_submit.py cnn_pred/svm.$event.test.model "cnn_vector/" $feat_dim_cnn cnn_pred/${event}_test_cnn.lst || exit 1;
  # python test_svm_submit.py cnn_pred/svm.$event.model "cnn_vector/" $feat_dim_cnn cnn_pred/${event}_test_cnn.lst || exit 1;
  # compute the average precision by calling the mAP package
  # ap ${event}_val_label cnn_pred/${event}_val_test_cnn.lst
done
