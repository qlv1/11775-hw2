This repository includes the HW2 submission for 11775.

The project consists of 4 parts: 1) feature extraction, 2) subsampling, 3) model training 4) model validation and testing

Part 1: Feature extraction.  
-- python surf_feat_extraction.py list/all.video config.yaml
-- python cnn_feat_extraction.py list/all.video config.yaml

Output features are stored in /surf_vector and /cnn_vector.

Part 2: Subsampling.
-- python select_frames.py all_trn.lst 100 select.surf.csv
-- python select_frames_cnn.py all_trn.lst 60 select.cnn.csv

Output files are stored as select.MODEL.csv.

Part 3: Training K-means.

-- python train_kmeans.py select.surf.csv 1000 kmeans.surf.1000.model
-- python train_kmeans.py select.cnn.csv 50 kmeans.cnn.50.model

Trained models are stored as kmeans.MODEL.n_clusters.csv.

Part 4: model validation and testing.
-- run.val.sh
-- run.test.sh

Output predictions are saved in /surf_pred and /cnn_pred.   
# 11775-hw2
