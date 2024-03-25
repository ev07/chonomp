import sys
import os

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLars
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error

rootdir = '../'
sys.path.append(rootdir)

import data_opener
import models



class Estimator():
    # whether the model supports begin given a list (variable, lag)
    # to filter the MTS window with.
    support_feature_filtering = None
    #define if the model needs the target column to be in the input
    is_autoregressive = False
    
    def __init__(self, config, target):
        self.config = config
    
    def fit_predict(self, data_train, data_test):
        """retourne une series pandas correspondant aux valeurs estimés sur le jeu de donnée de test
         1) 
        """
        
        return None
    
    
    def compute_metrics(self, y_pred, y_true, align=True):
        if align:
            y_true = y_true.loc[y_pred.index]  # align time series
        statistics = dict()
        statistics["sse"] = np.sum((y_true - y_pred)**2)
        statistics["R2"] = 1-statistics["sse"]/np.std(y_true)**2/len(y_true)
        statistics["rmse"] = np.sqrt(statistics["sse"]/len(y_true))
        statistics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        statistics["mfe"] = np.mean(y_true - y_pred)
        return statistics     
    
    def get_metrics(self, fittedvalues, data_test):
        return self.compute_metrics(fittedvalues, data_test[self.target])
        
    def compute_BBCCV(self, y_pred, y_true, K=100, seed=0):
        #return a list of rows corresponding to the in/out metrics in different bootstrap folds.
        # note that our bbccv is sampling N-1 samples and not N compared to the original method.
        rng = np.random.default_rng(seed=seed)
        results = []
        for fold in range(K):
            insample_index = rng.choice(y_pred.index, size=len(y_pred)-1, replace=True)
            outsample_index = [x for x in y_pred.index if x not in insample_index]
            insample_p = y_pred.loc[insample_index]
            insample_t = y_true.loc[insample_index]
            outsample_p = y_pred.loc[outsample_index]
            outsample_t = y_true.loc[outsample_index]
            inmetrics = self.compute_metrics(insample_p, insample_t, align=False)
            outmetrics = self.compute_metrics(outsample_p, outsample_t, align=False)
            
            row = dict(("in_"+key, value) for key, value in inmetrics.items())
            row = {**row, "fold":fold}
            row = {**row, **(dict(("out_"+key, value) for key, value in outmetrics.items()))}
            results.append(row)
        return results



class ARDLModel(Estimator):
    support_feature_filtering = False
    is_autoregressive = True

    def __init__(self, config, target):
        self.model = models.ARDLModel(config, target)
    def fit_predict(self, data_train, data_test):
        self.model.fit(data_train)
        fittedvalues = self.model.fittedvalues(data_test)
        return fittedvalues


class SKLearnVectorized(Estimator):
    support_feature_filtering = True
    # Base class for vectorized approaches in sklearn.
    def fit_predict(self, data_train, data_test, selected_features=None):
        X, y, _ = self.prepare_data_vectorize(data_train, selected_features)
        self.model = self.model.fit(X,y)
        X, y, indexes = self.prepare_data_vectorize(data_test, selected_features)
        fittedvalues = self.model.predict(X)
        fittedvalues = pd.Series(fittedvalues, index=indexes)
        return fittedvalues
        
    def prepare_data_vectorize(self,data, selected_features = None):
        # used to vectorize several timesteps in a dimension 1 vector.
        # vectorize
        y = data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [data.values[i:i+self.lags].reshape((-1,)) for i in range(len(data)-self.lags)]
        X = np.array(window_X)
        # if selected features were given, build filter
        if selected_features is not None:
            mask = np.zeros((self.lags, len(data.columns)), dtype=bool)
            column_index = dict([(name, i) for i, name in enumerate(data.columns)])
            for name, lag in selected_features:
                if lag<=self.lags:  # only count lags in the self.lags window
                    mask[-lag,column_index[name]]=True
            mask = mask.reshape((-1,))
            # apply filter
            X = X[:,mask]
        
        return X, y, indexes
    
    def fit(self, data_train, selected_features=None):
        X, y, _ = self.prepare_data_vectorize(data_train, selected_features)
        self.model = self.model.fit(X,y)
    def predict(self, data_test, selected_features=None):
        X, y, indexes = self.prepare_data_vectorize(data_test, selected_features)
        fittedvalues = self.model.predict(X)
        fittedvalues = pd.Series(fittedvalues, index=indexes)
        return fittedvalues

class SVRModel(SKLearnVectorized):
    def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model =  SVR(**config["skconfig"])

class KNeighborsRegressorModel(SKLearnVectorized):
    #same fit_predict as SVRModel, so inheritance is easier.
    def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model = KNeighborsRegressor(**config["skconfig"])
        
class LassoLarsModel(SKLearnVectorized):
     def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model = LassoLars(**config["skconfig"])
        
        
##################################################################
#                                                                #
#   Create configs for completion and optuna                     #
#                                                                #
##################################################################

def complete_config_from_parameters(name, hyperparameters):
    if name == "ARDLModel":
        config = { "constructor" : {"lags":hyperparameters.get("lags", 10),
                                    "order":hyperparameters.get("order", 10),
                                    "causal":True,
                                    "trend":hyperparameters.get("trend", "ct"),
                                    "seasonal":hyperparameters.get("seasonal", False),
                                    "period":hyperparameters.get("period", None),
                                    "missing":"drop"},
                   "fit" : {"cov_type":hyperparameters.get("cov_type", "HC0"),
                            "cov_kwds":hyperparameters.get("cov_kwds", None)}
                 }
    elif name == "SVRModel":
        config = {"lags":hyperparameters.get("lags", 10),
                  "skconfig":{"kernel":hyperparameters.get("kernel", "rbf"),
                              "degree":hyperparameters.get("degree", 3),
                              "gamma":hyperparameters.get("gamma", "scale"),
                              "coef0":hyperparameters.get("coef0", 0.),
                              "tol":hyperparameters.get("tol", 0.001),
                              "C":hyperparameters.get("C", 1.0),
                              "epsilon":hyperparameters.get("epsilon", 0.1),
                              "shrinking":True}}
    elif name == "KNeighborsRegressorModel":
        config = {"lags":hyperparameters.get("lags", 10),
                  "skconfig":{"n_neighbors":hyperparameters.get("n_neighbors", 5),
                              "weights":hyperparameters.get("weights", "uniform"),
                              "algorithm":hyperparameters.get("algorithm", "auto"),
                              "leaf_size":hyperparameters.get("leaf_size", 30),
                              "p":hyperparameters.get("p", 2),
                              "metric":"minkowski"}}
    elif name == "LassoLarsModel":
        config = {"lags":hyperparameters.get("lags", 10),
                  "skconfig":{"alpha":hyperparameters.get("alpha", 1.),
                              "fit_intercept":True,
                              "fit_path":False}}
    return config

def generate_optuna_parameters(name, trial):
    hp = dict()
    if name == "ARDLModel":
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["order"] = hp["lags"]
        hp["trend"] = trial.suggest_categorical("trend",["n","t","c", "ct"])
    elif name == "SVRModel":
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["kernel"] = trial.suggest_categorical("kernel",["linear","rbf","poly", "sigmoid"])
        #hp["degree"] = trial.suggest_int("degree",2,5,1,log=False)
        hp["coef0"] = trial.suggest_float("coef0", 0.0, 2.)
        hp["C"] = trial.suggest_float("C", 0.05, 20., log=True)
    elif name == "KNeighborsRegressorModel":
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["n_neighbors"] = trial.suggest_int("n_neighbors",3,20,1,log=True)
        hp["weights"] = trial.suggest_categorical("weights",["uniform", "distance"])
        hp["leaf_size"] = trial.suggest_int("leaf_size",20,50,1,log=True)
        hp["p"] = trial.suggest_int("p", 1, 2, 1, log=False)
    elif name == "LassoLarsModel":
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["alpha"] = trial.suggest_float("alpha", 0.001, 10., log=True)
    return hp


def generate_optuna_search_space(name):
    hp = dict()
    if name == "ARDLModel":
        hp["lags"] = [2]
        hp["trend"] = ["n","ct"]
    elif name == "SVRModel":
        hp["lags"] = [20]
        hp["kernel"] = ["rbf", "sigmoid"]
        hp["coef0"] = [0.0]
        hp["C"] = [ 0.1, 1., 10.]
    elif name == "KNeighborRegressorModel":
        hp["lags"] = [20]
        hp["n_neighbors"] = [5,  10,  50]
        hp["weights"] = trial.suggest_categorical("weights",["uniform", "distance"])
        hp["leaf_size"] = [20, 50]
        hp["p"] = [ 1, 2]
    elif name == "LassoLarsModel":
        hp["lags"] = [20]
        hp["alpha"] = [0.001,0.01, 0.1,  1.,  10.]
    return hp



