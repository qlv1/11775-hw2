import numpy as np

if __name__ == '__main__':
    fread = open('all_val.lst',"r")
    X_list = np.empty
    i = 0
    for line in fread.readlines():
        mfcc_name, label = line.split(' ')
        label = label.replace('\n','')
        if label == 'P001':
            X = np.array([1, 0, 0])
        elif label == 'P002':
            X = np.array([0, 1, 0])
        elif label == 'P003':
            X = np.array([0, 0, 1])
        else:
            X = np.array([0, 0, 0])

        if i == 0:
            X_list = X
        else:
            X_list = np.vstack((X_list, X))

        i = 1

    print (X_list[:, 0].astype(int))
    print (X_list[:, 1].astype(int))
    print (X_list[:, 2].astype(int))
    np.savetxt('P001_val_label', X_list[:, 0].astype(int))
    np.savetxt('P002_val_label', X_list[:, 1].astype(int))
    np.savetxt('P003_val_label', X_list[:, 2].astype(int))
