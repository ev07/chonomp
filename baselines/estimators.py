import sys
import os

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

rootdir = '../'
sys.path.append(rootdir)

import data_opener
import models



class Estimator():
    def compute_metrics(self, y_pred, y_true):
        y_true = y_true[y_pred.index]  # align time series
        
        statistics = dict()
        statistics["sse"] = np.sum((y_true - y_pred)**2)
        statistics["R2"] = 1-statistics["sse"]/np.std(y_true)**2/len(y_true)
        statistics["rmse"] = np.sqrt(statistics["sse"]/len(y_true))
        statistics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        statistics["mfe"] = np.mean(y_true - y_pred)
        return statistics
    
    def get_metrics(self, fittedvalues, data_test):
        return self.compute_metrics(fittedvalues, data_test[self.target]


class ARDLModel(Estimator):
    def __init__(self, config, target):
        self.model = models.ARDLModel(config, target)
    def fit_predict(self, data_train, data_test):
        self.model.fit(data_train)
        fittedvalues = self.model.fittedvalues(data_test)
        return fittedvalues

class SKLearnVectorized(Estimator):
    # Base class for vectorized approaches in sklearn.
    def fit_predict(self, data_train, data_test):
        X, y, _ = self.prepare_data_vectorize(data_train)
        self.model = self.model.fit(X,y)
        X, y, indexes = self.prepare_data_vectorize(data_test)
        fittedvalues = self.model.predict(X)
        fittedvalues = pd.Series(fittedvalues, index=indexes)
        return fittedvalues
        
    def prepare_data_vectorize(self,data):
        # used to vectorize several timesteps in a dimension 1 vector.
        y = data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [data.values[i:i+self.lags].reshape((-1,)) for i in range(len(data)-self.lags)]
        X = np.array(window_X)
        return X, y, indexes

class SVRModel(SKLearnVectorized):
    def __init__(self, config, target, lags):
        self.target = target
        self.lags = lags
        self.model =  SVR(**config)

class KNeighborsRegressorModel(SKLearnVectorized):
    #same fit_predict as SVRModel, so inheritance is easier.
    def __init__(self, config, target):
        self.target = target
        self.lags = lags
        self.model = KNeighborsRegressor(**config)
        

        
