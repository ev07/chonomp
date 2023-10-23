
from tsGOMP import tsGOMP_OneAssociation
from associations import PearsonMultivariate, SpearmanMultivariate
from models import ARDLModel

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
          
from statsmodels.tsa.vector_ar.var_model import VAR
          
class FeatureSelector():
    # whether the feature selection operates on variable (select columns from data): "variable"
    # or selects pairs (variable, lags): "variable, lag"
    selection_mode = None  

    def __init__(self, config, target):
        """
        Params:
        """
        self.config=config
        self.target = target
        
    def fit(self,data):
        pass
        
    def get_selected_features(self):
        return self.selected






class ChronOMP(FeatureSelector):

    selection_mode = "variable"  # the returned itemset consists in variables without lags.

    def __init__(self, config, target):
        super().__init__(config,target)
        config = self._config_init()
        self.instance = tsGOMP_OneAssociation(config, self.target)
        
    def _config_init(self):
        association_constructor = {"Pearson":PearsonMultivariate, "Spearman": SpearmanMultivariate}[self.config["association"]]
        association_config = self.config["association_config"]
        
        model_constructor = {"ARDL":ARDLModel}[self.config["model"]]
        model_config = self.config["model_config"]
        
        config = self.config["config"]
        config = {**config, "association": association_constructor,
                  "association.config": association_config,
                  "model": model_constructor,
                  "model.config": model_config}
        return config
        
    def fit(self, data):
        self.instance.fit(data)
    
    def get_selected_features(self):
        return self.instance.get_selected_features()
        
        
        
class ModifiedRFE(FeatureSelector):
    """
    Principle: Apply RFE on the vectorized data. We get a list of selected (variable, lags).
    Then, output the list of unique selected variables.
    """
    
    selection_mode = "variable"  # the returned itemset consists in variables without lags.
    
    def __init__(self, config, target):
        super().__init__(config,target)
        model = self._model_init()
    
    def _model_init(self):
        constructor = {"SVR":SVR}[self.config["model"]]
        self.model = constructor[self.config["model_config"]]
        return model
        
    
    def prepare_data_vectorize(self,data):
        # used to vectorize several timesteps in a dimension 1 vector.
        y = data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [data.values[i:i+self.lags].reshape((-1,)) for i in range(len(data)-self.lags)]
        X = np.array(window_X)
        return X, y, indexes
    
    def fit(self, data):
        X, y, _ = self.prepare_data_vectorize(data, self.config["lags"], self.target)
        
        rfe=RFE(estimator=self.model, n_features_to_select=self.config["n_features_to_select"], step=self.config["step"])
        rfe = rfe.fit(X, y)
        
        # if any lag was selected, select the whole feature
        indexes = np.any(np.array(rfe.support_).reshape((-1,len(data.columns))), axis=0)
        selected = data.columns[indexes]
        self.selected=selected
        return selected
        
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
    
    


##################################################################
#                                                                #
#   Create configs for completion and optuna                     #
#                                                                #
##################################################################

def complete_config_from_parameters(name, hyperparameters):
    if name == "ChronOMP":
        config = {"model": hyperparameters.get("model", "ARDL"),
                  "model_config": 
                     { "constructor" : {"lags":hyperparameters.get("lags", 10),
                                        "order":hyperparameters.get("order", 10),
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
    elif name == "ModifiedRFE" or name == "RFE":
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
    elif name == "BivariateGranger":
        config = {"maxlags": hyperparameters.get("maxlags", 10),
                  "alpha_level": hyperparameters.get("alpha_level", 0.05)}  
    return config
    
    
def generate_optuna_parameters(name, trial):
    hp = dict()
    if name == "ChronOMP":
        hp["model"] = "ARDL"
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["order"] = hp["lags"]
        hp["trend"] = trial.suggest_categorical("trend",["n","t","c", "ct"])
        hp["association"] = trial.suggest_categorical("association",["Pearson","Spearman"])
        hp["significance_threshold"] = trial.suggest_categorical("significance_threshold",[0.001, 0.005, 0.01, 0.05, 0.1])
        hp["method"] = trial.suggest_categorical("method",["f-test", "wald-test", "lr-test"])
        hp["max_features"] = trial.suggest_int("max_features", 5, 50, log=True)
        hp["valid_obs_param_ratio"] = trial.suggest_categorical("valid_obs_param_ratio",[1., 5., 10.])
    elif name == "ModifiedRFE" or name == "RFE":
        hp["model"] = "SVR"
        hp["n_features_to_select"] = trial.suggest_int("n_features_to_select", 5, 50, log=True)
        hp["step"] = trial.suggest_int("step", 4, 10, log=False)
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["kernel"] = trial.suggest_categorical("kernel",["linear","rbf","poly", "sigmoid"])
        hp["degree"] = trial.suggest_int("degree",2,5,1,log=False)
        hp["coef0"] = trial.suggest_float("coef0", 0.01, 2., log=True)
        hp["C"] = trial.suggest_float("C", 0.05, 20.,log=True)
    elif name == "BivariateGranger":
        hp["maxlags"] = trial.suggest_int("maxlags",5,20,1,log=False)
        hp["alpha_level"] = trial.suggest_float("alpha_level",0.0001,  0.1, log=True)
    return hp

    
    


