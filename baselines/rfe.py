import sys
import os


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from scipy.stats import rankdata
import numpy as np
import pandas as pd

rootdir = '../'
sys.path.append(rootdir)

import data_opener

def prepare_data(data_train, lags, target):
    y = data_train[target].values[lags:]
    window_X = [data_train.values[i:i+lags].reshape((-1,)) for i in range(len(data_train)-lags)]
    X = np.array(window_X)
    return X, y

def get_features(data_train, target, lags, stop_features, step=None, verbose=0):
    if step is None:
        step = lags
        
    X, y = prepare_data(data_train, lags, target)
    svr = SVR(kernel="linear")
    rfe=RFE(estimator=svr, n_features_to_select=stop_features, step=step, verbose=verbose)
    rfe = rfe.fit(X, y)
    
    # if any lag was selected, select the whole feature
    indexes = np.any(np.array(rfe.support_).reshape((-1,len(data_train.columns))),axis=0)
    selected = data_train.columns[indexes]
    
    # the ranking of a feature is the minimum of the ranking of any lag of this feature.
    rankings = np.min(np.array(rfe.ranking_).reshape((-1, len(data_train.columns))),axis=0)
    rankings = np.array(rankdata(rankings)).astype(int)-1
    ranked = np.array(sorted(zip(rankings,np.array(data_train.columns))))[:,1]
    
    return selected, ranked







if __name__=="__main__":
    
    dataset = "SynthNonlin/7ts2h"
    filename = "data_0.csv"
    data, var_names, causes = data_opener.open_dataset_and_ground_truth(dataset, filename, rootdir=rootdir)
    
    data_train = data[:int(len(data)*0.7)]
    
    selected, ranked = get_features(data_train, "A", 10, 4, verbose=1)
    
    print(selected)
    print(ranked)
