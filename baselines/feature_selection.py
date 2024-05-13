import numpy as np

from tsGOMP import tsGOMP_OneAssociation, tsGOMP_train_val
from associations import PearsonMultivariate, SpearmanMultivariate, LinearPartialCorrelation, ModelBasedPartialCorrelation
from models import ARDLModel, SVRModel

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LassoLars
from sklearn.feature_selection import SelectFromModel
          
from statsmodels.tsa.vector_ar.var_model import VAR

import group_lasso
import mrmr

#from baselines.SyPI import SyPI_method

#  Error class for the SelectFromModel instance, where giving a maximal number of selected variables above the size of the data is impossible.

class MaximalSelectedError(ValueError):
    def __init__(self, maxparam, datasize):
        self.message = "SelectFromModel was given max_features = {} but the number of columns in the data is {}.".format(maxparam, datasize)
        super().__init__(self.message)

##################################################################
#                                                                #
#               Feature Selection Base class                     #
#                                                                #
##################################################################

class FeatureSelector():
    # whether the feature selection operates on variable (select columns from data): "variable"
    # or selects pairs (variable, lags): "variable, lag"
    # unless required explicitely, should always be 'variable'
    selection_mode = None  

    def __init__(self, config, target):
        """
        Params:
            config: any dictionnary containing the configuration (hyperparameters) of the FS algorithm
            target: str, name of the target column
        """
        self.config=config
        self.target = target
        
    def fit(self,data):
        """
        Params:
            data: pd.DataFrame, pandas dataframe with time in lines and attributes in columns.
            
        To be implemented by each method
        Returns:
            None, or the same thing as get_selected_features if you want
        """
        pass
        
    def get_selected_features(self):
        """
        To be implemented by each method
        Returns:
            selected: The list of selected features as column names in the original dataframe
        """
        pass

    def prepare_data_vectorize(self, data, lags, groups=False):
        """
        From a pandas dataframe with time in lines and attributes in columns, 
        create a windowed version where each (variable, lag) is a column and lag<lags.
        
        Params:
            data: pd.DataFrame, the pandas dataframe of the data
            lags: int, the number of lags in the window (window size)
            groups: bool, if true, returns the groups.
        Returns:
            X: np.array, the new predictor matrix
            y: np.array, the new predicted value vector
            indexes: the original indexes of the predicted value (on which we can use pd.DataFrame(y, index=indexes))
            (optional) groups: np.array, the index of each column of data corresponding to each column of X
        """
        self.lags = lags
        # used to vectorize several timesteps in a dimension 1 vector.
        y = data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [data.values[i:i+self.lags].reshape((-1,)) for i in range(len(data)-self.lags)]
        X = np.array(window_X)
        if groups: 
            group_names = list(range(len(data.columns)))*self.lags
            return X, y, indexes, group_names
        return X, y, indexes

    def vector_mask_to_columns(self, mask, data):
        """
        Given a mask (a vector of boolean with True if a feature is selected and False otherwise),
        covering the vectorized feature space (aka, each columns at each lag up to the window size),
        extract each column for which at least one lag was selected by the FS method.
        
        Params:
            mask: 1D np.array of bool, at True if the given vectorized feature is selected, False otherwise.
            data: pd.DataFrame, the data in the original format.
        Returns:
            selected: list of str, the list of column names in the original dataframe that were selected
        """
        indexes = np.any(np.array(mask).reshape((-1,len(data.columns))), axis=0)
        selected = data.columns[indexes]
        self.selected=list(selected)
        return self.selected
        
    def _complete_config_from_parameters(hyperparameters):
        """
        Static method transforming a flat hyperparameter dictionary into a valid configuration for a class object.
        Params:
            hyperparameters: dictionary, containing the hyperparameters
        Returns:
            config: dictionary, containing a valid configuration
        """
        pass
    
    def _generate_optuna_parameters(trial):
        """
        Static method sampling hyperparametrers from an optuna trial.
        Params:
            trial: optuna.trial.Trial, the trial object with which to sample the parameters
        Returns:
            hp: the hyperparameter dictionary indexed by parameter name, containing each associated sampled value.
        """
        pass
        
    def _generate_optuna_search_space():
        """
        Static method defining the search space for an optuna GridSampler optimizer.
        Returns:
            hp: the grid space dictionary indexed by parameter name, containing for each parameter a list of values to try.
        """        
        pass
     
     
##################################################################
#                                                                #
#                   Our proposed algorithm                       #
#                                                                #
##################################################################  
        

class ChronOMP(FeatureSelector):

    selection_mode = "variable"  # the returned itemset consists in variables without lags.

    def __init__(self, config, target, verbosity=0):
        super().__init__(config,target)
        config = self._config_init()
        self.instance = tsGOMP_OneAssociation(config, self.target, verbosity=verbosity)
        
    def _config_init(self):
        association_constructor = {"Pearson":PearsonMultivariate, "Spearman": SpearmanMultivariate}[self.config["association"]]
        association_config = self.config["association_config"]
        
        model_constructor = {"ARDL":ARDLModel}[self.config["model"]]
        model_config = self.config["model_config"]
        
        config = self.config["config"]
        config = {**config, "association": association_constructor,
                  "association.config": association_config,
                  "model": model_constructor,
                  "model.config": model_config,
                  "equivalent_version":"f"}
        return config
        
    def fit(self, data):
        self.instance.fit(data)
    
    def get_selected_features(self):
        return self.instance.get_selected_features()
    
    def _complete_config_from_parameters(hyperparameters):
        config = {"model": hyperparameters.get("model", "ARDL"),
                  "model_config": 
                     { "constructor" : {"lags":hyperparameters.get("lags", 10),
                                        "order":hyperparameters.get("order", hyperparameters.get("lags", 10)),
                                        "causal":True,
                                        "trend":hyperparameters.get("trend", "ct"),
                                        "seasonal":hyperparameters.get("seasonal", False),
                                        "period":hyperparameters.get("period", None),
                                        "missing":"drop"},
                       "fit" : {"cov_type":hyperparameters.get("cov_type", "HC0"),
                                "cov_kwds":hyperparameters.get("cov_kwds", None)}
                     },
                 "association": hyperparameters.get("association", "Pearson"),
                 "association_config":{
                       "return_type": hyperparameters.get("return_type", "p-value"),
                       "lags": hyperparameters.get("lags", 10),
                       "selection_rule": hyperparameters.get("selection_rule", "max"),
                     },
                 "config":
                     { "significance_threshold": hyperparameters.get("significance_threshold", 0.05),
                       "method": hyperparameters.get("method", "f-test"),
                       "max_features": hyperparameters.get("max_features", 5),
                       "valid_obs_param_ratio": hyperparameters.get("valid_obs_param_ratio", 10),
                       "choose_oracle": False
                     }
                 }
        return config
    
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["model"] = "ARDL"
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["order"] = hp["lags"]
        hp["trend"] = trial.suggest_categorical("trend",["n","t","c", "ct"])
        hp["association"] = trial.suggest_categorical("association",["Pearson","Spearman"])
        hp["significance_threshold"] = trial.suggest_float("significance_threshold", 1e-20, 0.1, log=True)
        hp["method"] = trial.suggest_categorical("method",["f-test", "wald-test", "lr-test"])
        hp["max_features"] = trial.suggest_int("max_features", 5, 50, log=True)
        hp["valid_obs_param_ratio"] = trial.suggest_categorical("valid_obs_param_ratio",[1., 5., 10.])
        return hp
        
    def _generate_optuna_search_space():
        hp = dict()
        hp["model"] = ["ARDL"]
        hp["lags"] = [20]
        hp["trend"] = ["n","ct"]
        hp["association"] = ["Pearson"]#,"Spearman"]
        hp["significance_threshold"] = [1e-20, 1e-10, 1e-5, 1e-4]
        hp["method"] = ["f-test", "lr-test"]
        hp["max_features"] = [50]
        hp["valid_obs_param_ratio"] = [1.]
        return hp

class BackwardChronOMP(ChronOMP):
    def _config_init(self):
        config = super()._config_init()
        config["equivalent_version"]="fb"
        return config
    def fit(self, data):
        self.instance.fit(data)
    def _complete_config_from_parameters(hyperparameters):
        config = ChronOMP._complete_config_from_parameters(hyperparameters)
        config["config"]["significance_threshold_backward"] = hyperparameters.get("significance_threshold_backward", 0.05)
        config["config"]["method_backward"] = hyperparameters.get("method_backward", hyperparameters.get("method","f-test"))
        return config
    def _generate_optuna_parameters(trial):
        hp = ChronOMP._generate_optuna_parameters(trial)
        hp["significance_threshold_backward"] = trial.suggest_float("significance_threshold_backward", 1e-20, 0.1, log=True)
        hp["method_backward"] = trial.suggest_categorical("method",["f-test", "wald-test", "lr-test"])
        return hp
    def _generate_optuna_search_space():
        hp = ChronOMP._generate_optuna_search_space()
        hp["lags"] = [10]
        hp["method"] = ["f-test"]
        hp["significance_threshold"] = [1e-20, 1e-10, 1e-5, 1e-2]
        hp["significance_threshold_backward"] = [1e-20, 1e-10, 1e-5, 1e-2]
        hp["method_backward"] = ["lr-test"]
        return hp

class MultiSetChronOMP(ChronOMP):
    def __init__(self, config, target, equivalent_version="fg"):
        self.config=config
        self.target = target
        self.equivalent_version = version
        config = self._config_init()
        self.instance = tsGOMP_OneAssociation(config, self.target, verbosity=verbosity)
        
    def _config_init(self):
        association_constructor = {"Pearson":PearsonMultivariate, "Spearman": SpearmanMultivariate}[self.config["association"]]
        association_config = self.config["association_config"]
        
        model_constructor = {"ARDL":ARDLModel}[self.config["model"]]
        model_config = self.config["model_config"]
        
        partial_constructor = {"par_corr":LinearPartialCorrelation, 
                               "model_par_corr":ModelBasedPartialCorrelation
                               }[self.config["partial_correlation"]]
        partial_config = self.config["partial_correlation.config"]
        
        config = self.config["config"]
        config = {**config, "association": association_constructor,
                  "association.config": association_config,
                  "model": model_constructor,
                  "model.config": model_config,
                  "partial_correlation": partial_constructor,
                  "partial_correlation.config": partial_config,
                  "equivalent_version":self.equivalent_version,
                  }
        return config
    def _complete_config_from_parameters(hyperparameters):
        config = ChronOMP._complete_config_from_parameters(hyperparameters)
        config["config"]["equivalence_threshold"] = hyperparameters.get("equivalence_threshold", 0.05)
        config["partial_correlation"] = hyperparameters.get("partial_correlation", "model_par_corr")
        config["partial_correlation.config"] = {
                       #"method": hyperparameters.get("parr_corr_method", "pearson"),
                       "lags": hyperparameters.get("lags", 10),
                       #"selection_rule": hyperparameters.get("parr_corr_selection_rule", "min"),
                       "large_sample": hyperparameters.get("large_sample", False)
        }
        return config
    def _generate_optuna_parameters(trial):
        hp = ChronOMP._generate_optuna_parameters(trial)
        hp["equivalence_threshold"] = trial.suggest_float("equivalence_threshold", 0.00001, 0.1, log=True)
        hp["partial_correlation"] = trial.suggest_categorical("partial_correlation",["par_corr", "model_par_corr"])
        #hp["parr_corr_method"] = trial.suggest_categorical("parr_corr_method", ["pearson", "spearman"])
        #hp["parr_corr_selection_rule"] = trial.suggest_categorical("parr_corr_selection_rule", ["mean", "min"])
        hp["large_sample"] = trial.suggest_categorical("large_sample",[True, False])
        return hp
    def _generate_optuna_search_space():
        hp = ChronOMP._generate_optuna_search_space()
        hp["equivalence_threshold"] = [0.0001, 0.001, 0.05]
        hp["partial_correlation"] = ["model_par_corr"]
        #hp["parr_corr_method"] = ["pearson", "spearman"]
        #hp["parr_corr_selection_rule"] = ["min"]
        hp["large_sample"]=[False]
        return hp





class TrainTestChronOMP(ChronOMP):
    def __init__(self, config, target):
        self.config=config
        self.target = target
        config = self._config_init()
        self.instance = tsGOMP_train_val(config, self.target)
        
    def _config_init(self):
        association_constructor = {"Pearson":PearsonMultivariate, "Spearman": SpearmanMultivariate}[self.config["association"]]
        association_config = self.config["association_config"]
        
        model_constructor = {"SVR":SVRModel}[self.config["model"]]
        model_config = self.config["model_config"]
        
        config = self.config["config"]
        config = {**config, "association": association_constructor,
                  "association.config": association_config,
                  "model": model_constructor,
                  "model.config": model_config,
                  "equivalent_version": "f"}
        return config

    def _complete_config_from_parameters(hyperparameters):
        config = {"model": hyperparameters.get("model", "SVR"),
                  "model_config": 
                     {"lags": hyperparameters.get("lags", 10),
                      "skconfig":{"kernel":hyperparameters.get("kernel", "rbf"),
                              "degree":hyperparameters.get("degree", 3),
                              "gamma":hyperparameters.get("gamma", "scale"),
                              "coef0":hyperparameters.get("coef0", 0.),
                              "tol":hyperparameters.get("tol", 0.001),
                              "C":hyperparameters.get("C", 1.0),
                              "epsilon":hyperparameters.get("epsilon", 0.1),
                              "shrinking":True}
                     },
                 "association": hyperparameters.get("association", "Pearson"),
                 "association_config":{
                       "return_type": hyperparameters.get("return_type", "p-value"),
                       "lags": hyperparameters.get("lags", 10),
                       "selection_rule": hyperparameters.get("selection_rule", "max"),
                     },
                 "config":
                     { "significance_threshold": hyperparameters.get("significance_threshold", 0.0),
                       "method": hyperparameters.get("method", "rmse_diff"),
                       "max_features": hyperparameters.get("max_features", 5),
                       "valid_obs_param_ratio": hyperparameters.get("valid_obs_param_ratio", 10),
                       "choose_oracle": False,
                       "validation_ratio":0.1
                     }
                 }
        return config
    
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["model"] = "SVR"
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["kernel"] = trial.suggest_categorical("kernel",["linear","rbf","poly", "sigmoid"])
        hp["coef0"] = trial.suggest_float("coef0", 0.0, 2.)
        hp["C"] = trial.suggest_float("C", 0.05, 20., log=True)
        hp["association"] = trial.suggest_categorical("association",["Pearson","Spearman"])
        hp["significance_threshold"] = trial.suggest_float("significance_threshold", -0.05, 0., log=False)
        hp["method"] = trial.suggest_categorical("method",["rmse_diff"])
        hp["max_features"] = trial.suggest_int("max_features", 5, 50, log=True)
        hp["valid_obs_param_ratio"] = trial.suggest_categorical("valid_obs_param_ratio",[1., 5., 10.])
        return hp
        
    def _generate_optuna_search_space():
        hp = dict()
        hp["model"] = ["SVR"]
        hp["lags"] = [20]
        hp["kernel"] = ["rbf", "sigmoid"]
        hp["coef0"] = [0.0]
        hp["C"] = [ 0.1, 1., 10.]
        hp["association"] = ["Pearson","Spearman"]
        hp["significance_threshold"] = [-0.05, -0.01, 0.0]
        hp["method"] = ["rmse_diff"]
        hp["max_features"] = [50]
        hp["valid_obs_param_ratio"] = [1.]
        return hp

##################################################################
#                                                                #
#                       Lasso with LARS                          #
#                                                                #
##################################################################   


class VectorLassoLars(FeatureSelector):
    """
    Principle: Apply LassoLars on the vectorized data with a feature importance selector.
    We get a list of selected (variable, lags).
    Then, output the list of unique selected variables.
    """
    
    selection_mode = "variable"  # the returned itemset consists in variables without lags.
    
    def __init__(self, config, target):
        super().__init__(config,target)
        self._model_init()
    
    def _model_init(self):
        self.model = LassoLars(**self.config["model_config"])
    
    def fit(self, data):
        X, y, _ = self.prepare_data_vectorize(data, self.config["lags"])
        

        # There can be an error if the number of feature given in argument is too large.
        # Default strategy is to replace the parameter by the size of the feature space.
        # A parameter can be given explicitely to raise an error in a custom class instead.
        # This is where we handle it, with a custom error that we send back.
        max_features = self.config["max_features"]
        if max_features > X.shape[1] and not self.config.get("raiseMaximalSelectedError",False):
            max_features = X.shape[1]
        elif max_features > X.shape[1]:
            raise MaximalSelectedError(max_features, X.shape[1])
            
        FS=SelectFromModel(estimator=self.model, threshold=self.config["threshold"], max_features=max_features)
        FS = FS.fit(X, y)
        
        mask = FS.get_support(indices=False)
        selected = self.vector_mask_to_columns(mask, data)
        
        return selected

    def get_selected_features(self):
        return self.selected

    def _complete_config_from_parameters(hyperparameters):
        config = {"lags": hyperparameters.get("lags", 10),
                  "max_features": hyperparameters.get("max_features",10),
                  "threshold": hyperparameters.get("threshold",0.0001),
                  "model_config": {"alpha":hyperparameters.get("alpha", 1.),
                                   "fit_intercept":True,
                                   "fit_path":False}
                  }
        return config
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["max_features"] = trial.suggest_int("max_features", 5, 50, log=True)
        hp["threshold"] = trial.suggest_float("threshold", 0.000001, 0.01, log=True)
        hp["alpha"] = trial.suggest_float("alpha", 0.001, 10., log=True)
        return hp

    def _generate_optuna_search_space():
        hp = dict()
        hp["lags"] = [20]
        hp["max_features"] = [50]
        hp["threshold"] = [0.000001, 0.00001,  0.0001,  0.001, 0.01]
        hp["alpha"] = [0.001, 0.01, 0.1, 1.,  10.]
        return hp


##################################################################
#                                                                #
#                       Grouped Lasso                            #
#                                                                #
##################################################################

class GroupLasso(FeatureSelector):
    selection_mode = "variable"
    
    def __init__(self, config, target):
        super().__init__(config,target)
        self.selected=None
    
    def _model_init(self, groups):
        self.model = group_lasso.GroupLasso(groups, **self.config["model_config"],
                                            supress_warning=True)
        
    def fit(self, data):
        X, y, _, groups = self.prepare_data_vectorize(data, self.config["lags"],groups=True)
        self._model_init(groups)
        self.model.fit(X,y)
        mask = self.model.sparsity_mask_
        selected = self.vector_mask_to_columns(mask, data)
        self.selected = selected
        return selected
    
    def get_selected_features(self):
        return self.selected
    
    def _complete_config_from_parameters(hyperparameters):
        config = {"lags": hyperparameters.get("lags", 10),
                  "model_config":{
                      "group_reg": hyperparameters.get("group_reg", 0.001),
                      "l1_reg": hyperparameters.get("l1_reg", 0.001),
                      "n_iter": hyperparameters.get("n_iter", 200),
                      "tol": hyperparameters.get("tol", 1e-5)
                      }
                 }
        return config
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["group_reg"] = trial.suggest_float("group_reg",1e-20,1,log=True)
        hp["l1_reg"] = trial.suggest_float("l1_reg",1e-20,1,log=True)
        return hp

    def _generate_optuna_search_space():
        hp = dict()
        hp["lags"] = [10]
        hp["group_reg"] = [0.00001, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001,  0.0014, 0.0018, 0.0022, 0.0026,0.003,  0.0034, 0.0038, 0.0042, 0.0046, 0.005, 0.007, 0.009, 0.05]
        hp["l1_reg"] = [1e-20]
        return hp



##################################################################
#                                                                #
#               Recursive Feature Elimination                    #
#                                                                #
##################################################################




class ModifiedRFE(FeatureSelector):
    """
    Principle: Apply RFE on the vectorized data. We get a list of selected (variable, lags).
    Then, output the list of unique selected variables.
    """
    
    selection_mode = "variable"  # the returned itemset consists in variables without lags.
    
    def __init__(self, config, target):
        super().__init__(config,target)
        self._model_init()
    
    def _model_init(self):
        constructor = {"SVR":SVR}[self.config["model"]]
        self.model = constructor(**self.config["model_config"])
        
    
    def fit(self, data):
        X, y, _ = self.prepare_data_vectorize(data, self.config["lags"])
        
        rfe=RFE(estimator=self.model, n_features_to_select=self.config["n_features_to_select"], step=self.config["step"])
        rfe = rfe.fit(X, y)
        
        mask = np.array(rfe.support_)
        selected = self.vector_mask_to_columns( mask, data)
        return selected
    
    def get_selected_features(self):
        return self.selected
    
    def _complete_config_from_parameters(hyperparameters):
        default_model_config = {"kernel":hyperparameters.get("kernel", "rbf"),
                              "degree":hyperparameters.get("degree", 3),
                              "gamma":hyperparameters.get("gamma", "scale"),
                              "coef0":hyperparameters.get("coef0", 0.),
                              "tol":hyperparameters.get("tol", 0.001),
                              "C":hyperparameters.get("C", 1.0),
                              "epsilon":hyperparameters.get("epsilon", 0.1),
                              "shrinking":True}
                               
        config = {"model": hyperparameters.get("model", "SVR"),
                  "model_config": hyperparameters.get("model_config", default_model_config),
                  "lags": hyperparameters.get("lags", 10),
                  "n_features_to_select": hyperparameters.get("n_features_to_select", 10),
                  "step": hyperparameters.get("step", 10)
                 }
        return config
    
    def _generate_optuna_parameters(trial):
        hp["model"] = "SVR"
        hp["n_features_to_select"] = trial.suggest_int("n_features_to_select", 5, 50, log=True)
        hp["step"] = trial.suggest_int("step", 4, 10, log=False)
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["kernel"] = trial.suggest_categorical("kernel",["linear","rbf","poly", "sigmoid"])
        hp["degree"] = trial.suggest_int("degree",2,5,1,log=False)
        hp["coef0"] = trial.suggest_float("coef0", 0.01, 2., log=True)
        hp["C"] = trial.suggest_float("C", 0.05, 20.,log=True)
        return hp
    
    def _generate_optuna_search_space():
        hp["lags"] = [5,10,15]
        hp["model"] = ["SVR"]
        hp["n_features_to_select"] = [1,2,3,4,5,6,7,8,10,15,20,30,40,50]
        hp["step"] = [2, 4, 6, 8, 10]
        hp["kernel"] = ["linear","rbf"]
        hp["degree"] = [2]
        hp["coef0"] = [0.01]
        hp["C"] = [0.01, 0.1, 1., 10.]
        return hp

class RFE(ModifiedRFE):
    """
    The original RFE method.
    """

    selection_mode = "variable, lag"
    
    def fit(self, data):
        X, y, _ = self.prepare_data_vectorize(data, self.config["lags"], self.target)
        
        rfe=RFE(estimator=self.model, n_features_to_select=self.config["n_features_to_select"], step=self.config["step"])
        rfe = rfe.fit(X, y)
        
        mask = np.array(rfe.support_).reshape((-1,len(data.columns)))
        selected=[]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]:
                    selected.append((data.columns[i],self.lags-j))
        self.selected=selected
        return selected



##################################################################
#                                                                #
#           Bivariate Granger pairwise test                      #
#                                                                #
##################################################################


class BivariateGranger(FeatureSelector):
    """
    Bivariate Granger feature selection with statsmodel VAR instances.
    """
    
    selection_mode = "variable"
    
    def __init__(self, config, target):
        super().__init__(config, target)
    
    def fit(self, data):
        signif = self.config["alpha_level"]
        selected = []
        for column in data.columns:
            if column==self.target:
                continue
            model = VAR(data[[self.target, column]])
            results = model.fit(maxlags=self.config["maxlags"])
            pvalue = results.test_causality(self.target, causing=column, signif=signif).pvalue
            if pvalue < signif:
                pvalue = results.test_causality(column, causing=self.target, signif=signif).pvalue
                if pvalue > signif:
                    selected.append(column)
        self.selected=selected
        return selected
    
    def get_selected_features(self):
        return self.selected    

    def _complete_config_from_parameters(hyperparameters):
        config = {"maxlags": hyperparameters.get("maxlags", 10),
                  "alpha_level": hyperparameters.get("alpha_level", 0.05)}  
        return config
        
    def _generate_optuna_parameters(trial):
        hp = dict()    
        hp["maxlags"] = trial.suggest_int("maxlags",5,20,1,log=False)
        hp["alpha_level"] = trial.suggest_float("alpha_level",0.0001,  0.1, log=True)
        return hp
    
    def _generate_optuna_search_space():
        hp = dict()
        hp["maxlags"] = [5,10,15]
        hp["alpha_level"] = [0.001, 0.01, 0.05,  0.1]
        return hp
        
##################################################################
#                                                                #
#           Vectorized MRMR                                      #
#                                                                #
##################################################################

class VectorMRMR(FeatureSelector):
    """
    Principle: Apply mRMR on the vectorized data.
    We get a list of selected (variable, lags).
    Then, output the list of unique selected variables.
    """
    
    selection_mode = "variable"  # the returned itemset consists in variables without lags.
    
    def __init__(self, config, target):
        super().__init__(config,target)
    
    def fit(self, data):
        X, y, _ = self.prepare_data_vectorize(data, self.config["lags"])
        X_df, y_series = pd.DataFrame(X), pd.Series(y)
        

        # There can be an error if the number of feature given in argument is too large.
        # Default strategy is to replace the parameter by the size of the feature space.
        # A parameter can be given explicitely to raise an error in a custom class instead.
        # This is where we handle it, with a custom error that we send back.
        num_features = self.config["num_features"]
        
        vector_selected = mrmr.pandas.mrmr_regression(
            X_df, y_series, num_features,
            cat_features=None,
            only_same_domain=False, return_scores=False,
            n_jobs=1, show_progress=False, **self.config["config"])
        
        mask = [var in vector_selected for var in X_df.columns]
        
        selected = self.vector_mask_to_columns(mask, data)
        
        return selected

    def get_selected_features(self):
        return self.selected

    def _complete_config_from_parameters(hyperparameters):
        config = {"lags": hyperparameters.get("lags", 10),
                  "num_features": hyperparameters.get("num_features",10),
                  "config": {"alpha":hyperparameters.get("relevance", "rf"),
                                   "redundancy":hyperparameters.get("redundancy", "c"),
                                   "denominator":hyperparameters.get("denominator", "mean")}
                  }
        return config
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["lags"] = trial.suggest_int("lags",5,50,1,log=False)
        hp["num_features"] = trial.suggest_int("num_features", 5, 100, log=True)
        hp["relevance"] = trial.suggest_categorical("relevance",["f","rf"])
        hp["denominator"] = trial.suggest_categorical("denominator",["mean","max"])
        return hp

    def _generate_optuna_search_space():
        hp = dict()
        hp["lags"] = [50]
        hp["num_features"] = [5, 10, 20, 30, 50, 70, 100]
        hp["relevance"] = ["rf"]
        hp["denominator"] = ["mean","max"]
        return hp

##################################################################
#                                                                #
#                       SyPI                                     #
#                                                                #
##################################################################



class SyPI(FeatureSelector):
    def __init__(self, config, target):
        super().__init__(config, target)
        self.p_cond1 = config["p_cond1"]
        self.p_cond2 = config["p_cond2"]
        self.threshold_lasso = config["threshold_lasso"]
        self.lags = config["lags"]
    
    
    def fit(self, data):
        # by default in SyPI, the target variable is the last one. We need to reorder the dataframe.
        columns = [name for name in data.columns if name!=self.target]
        columns = columns + [self.target]
        # also, the type of data expected by the function is an np.array, with covariates in rows and timesteps in columns.
        X = data[columns].values.T
        
        predicted_causes_indices = SyPI_method("scipy_lassoalgo",  # regression algorithm: only this choice implemented
                    "linear",  # partial correlation algorithm: only this choice implemented
                    False,  # do not normalize the data since this was done already as preprocessing
                    None,  # no lambda parameter in the lasso used, instead chosen with AIC
                    self.lags,
                    self.p_cond1, 
                    self.p_cond2, 
                    self.threshold_lasso,
                    None, None, None,  # no ground truth provided here
                    X)
        selected_column_names = list(np.array(columns)[predicted_causes_indices])
        self.selected = selected_column_names
        return self.selected
    
    def get_selected_features(self):
        return self.selected
        
    def _complete_config_from_parameters(hyperparameters):
        config = {"lags": hyperparameters.get("lags", 10),
                  "p_cond1": hyperparameters.get("p_cond1",0.001),
                  "p_cond2": hyperparameters.get("p_cond2",hyperparameters.get("p_cond1",0.001)),
                  "threshold_lasso": hyperparameters.get("threshold_lasso",0.001)}
        return config
        
    def _generate_optuna_parameters(trial):
        hp = dict()
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["threshold_lasso"] = trial.suggest_float("threshold_lasso",0.00001, 0.1, log=True)
        hp["p_cond1"] = trial.suggest_float("p_cond1",0.001,  0.1, log=True)
        hp["p_cond2"] = trial.suggest_float("p_cond2",0.001,  0.1, log=True)
        return hp
    
    def _generate_optuna_search_space():
        hp = dict()
        hp["lags"] = [5,10,15]
        hp["threshold_lasso"] = [0.000001, 0.00001,  0.0001,  0.001, 0.01]
        hp["p_cond1"] = [0.001, 0.01, 0.05, 0.1]
        hp["p_cond2"] = [0.001, 0.01, 0.05, 0.1]
        return hp
        

##################################################################
#                                                                #
#                       No Selection Baseline                    #
#                                                                #
##################################################################



class NoSelection(FeatureSelector):
    def fit(self, data):
        self.selected = list(data.columns)
        return self.selected
    
    def get_selected_features(self):
        return self.selected
        
    def _complete_config_from_parameters(hyperparameters):
        return dict()
        
    def _generate_optuna_parameters(trial):
        return dict()
    
    def _generate_optuna_search_space():
        return dict()

##################################################################
#                                                                #
#   Create configs for completion and optuna                     #
#                                                                #
##################################################################

def complete_config_from_parameters(name, hyperparameters):
    if name == "ChronOMP":
        config = ChronOMP._complete_config_from_parameters(hyperparameters)
    elif name == "BackwardChronOMP":
        config = BackwardChronOMP._complete_config_from_parameters(hyperparameters)
    elif name == "TrainTestChronOMP":
        config = TrainTestChronOMP._complete_config_from_parameters(hyperparameters)
    elif name == "ModifiedRFE" or name == "RFE":
        config = ModifiedRFE._complete_config_from_parameters(hyperparameters)
    elif name == "BivariateGranger":
        config = BivariateGranger._complete_config_from_parameters(hyperparameters)
    elif name == "VectorLassoLars":
        config = VectorLassoLars._complete_config_from_parameters(hyperparameters)
    elif name == "SyPI":
        config = SyPI._complete_config_from_parameters(hyperparameters)
    elif name == "GroupLasso":
        config = GroupLasso._complete_config_from_parameters(hyperparameters)
    elif name == "NoSelection":
        config = NoSelection._complete_config_from_parameters(hyperparameters)
    return config
    
    
def generate_optuna_parameters(name, trial):
    hp = dict()
    if name == "ChronOMP":
        hp = ChronOMP._generate_optuna_parameters(trial)
    elif name == "BackwardChronOMP":
        hp = BackwardChronOMP._generate_optuna_parameters(trial)
    elif name == "TrainTestChronOMP":
        hp = TrainTestChronOMP._generate_optuna_parameters(trial)
    elif name == "ModifiedRFE" or name == "RFE":
        hp = ModifiedRFE._generate_optuna_parameters(trial)
    elif name == "BivariateGranger":
        hp = BivariateGranger._generate_optuna_parameters(trial)
    elif name == "VectorLassoLars":
        hp = VectorLassoLars._generate_optuna_parameters(trial)
    elif name == "SyPI":
        hp = SyPI._generate_optuna_parameters(trial)
    elif name == "GroupLasso":
        hp = GroupLasso._generate_optuna_parameters(trial)
    elif name == "NoSelection":
        hp = NoSelection._generate_optuna_parameters(trial)
    return hp

    
def generate_optuna_search_space(name):
    hp = dict()
    if name == "ChronOMP":
        hp = ChronOMP._generate_optuna_search_space()
    elif name == "BackwardChronOMP":
        hp = BackwardChronOMP._generate_optuna_search_space()
    elif name == "TrainTestChronOMP":
        hp = TrainTestChronOMP._generate_optuna_search_space()
    elif name == "ModifiedRFE" or name == "RFE":
        hp = ModifiedRFE._generate_optuna_search_space()
    elif name == "BivariateGranger":
        hp = BivariateGranger._generate_optuna_search_space()
    elif name == "VectorLassoLars":
        hp = VectorLassoLars._generate_optuna_search_space()
    elif name == "SyPI":
        hp = SyPI._generate_optuna_search_space()
    elif name == "GroupLasso":
        hp = GroupLasso._generate_optuna_search_space()
     elif name == "NoSelection":
        hp = NoSelection._generate_optuna_search_space()
    return hp


