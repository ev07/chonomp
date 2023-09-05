from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults, AutoRegResultsWrapper
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from scipy.stats import f as fdistrib, wilcoxon, chi2

from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd
import pydash

#from modelsources import deepAR

from torch import device
import torch.cuda



# custom error for ARDL model

class NotEnoughDataError(ValueError):
    def __init__(self, datasize, lags, orders):
        self.message = "Model needs more lags of the data than provided to make predictions.\nPlease consider increasing the test data size.\nData length: {}, number of lags: {}, number of orders: {}".format(datasize, lags, orders)
        super().__init__(self.message)
    


##
#
#   Learning wrappers
#
##


class LearningModel:
    def __init__(self, config, target):
        self.config = config
        self.target = target
        
        self.data = None
        self.model = None

    #
    # part that need to be implemented for each learning model    
    #
    
    def fit(self, data):
        pass

    def fittedvalues(self, data=None):
        # return the fitted values of the model.
        # should be a pd.Series with corresponding index to original data
        # the series should not contain NaN timestamps.
        pass
        
    def stopping_metric(self, previous_model, method):
        # should return a metric that corresponds more or less to p-values.
        # the lower, the more incentive to keep adding new variables to the selected set
        pass
    
    def has_too_many_parameters(self):
        # part of the stopping criterion: verify if there are 10 times more timestamps in the data
        # than parameters in the model.
        pass
        
    #
    # part that can be let as-is or modified after inheritance.
    #
    
    def residuals(self, data=None):
        # output should be a pd.DataFrame, rows index should correspond to the original data
        # the series should not contain NaN timestamps.
        if data is None:
            data = self.data
        fittedvalues = self.fittedvalues(data)
        targetdata = data[self.target]
        targetdata = targetdata[fittedvalues.index]
        residuals = targetdata - fittedvalues
        # make the results a dataframe adressable by target.
        df = pd.DataFrame({self.target: residuals})
        return df

    def sse(self, data=None):
        """
        Compute the Sum of Squared Errors of the residuals of the target series.
        """
        residuals = self.residuals(data)[self.target]
        return pd.Series.sum(residuals**2)

    def total_variation(self, data=None):
        """
        Compute the Total Variation of the target series, for all timestamps for which residuals are computed.
        Total Variation = len(series) * series.std()**2
        """
        indexes = self.residuals(data).index  # only take timestamps over which residuals exist
        if data is None:
            data = self.data
        originals = data.loc[indexes][self.target]
        return len(indexes)*(originals.std()**2)

    def statistics(self, data=None):
        # how to get statistics for the training set, or a test set.
        # current global statistics are sse, rmse, R2, mfe, mape. 
        # They can be obtained only from the data and the fitted values.
        if data is None:
            data = self.data
        
        statistics = dict()
        residuals = self.residuals(data)
        statistics["sse"] = self.sse(data)
        statistics["rmse"] = np.sqrt(statistics["sse"]/len(residuals))
        statistics["R2"] = 1 - statistics["sse"]/self.total_variation(data)
        statistics["mfe"] = np.mean(residuals[self.target])
        
        fittedvalues = self.fittedvalues(data)
        rangefitted = fittedvalues.index  # use only timestamps for which we have outputs
        statistics["mape"] = mean_absolute_percentage_error(data.loc[rangefitted][self.target], fittedvalues)
        
        return statistics




class VARModel(LearningModel):
    """
    Adapted from statsmodels.tsa.api.VAR.
     - the config parameter must be accepted by the VAR method of the statsmodel package.
     - uses lag estimation with specified criterion.
     - target variable is included in the observed variables
     - uses minus aic of final model as significance

    Univariate case is covered by statsmodels.tsa.ar_model.AutoReg
    """
    def __init__(self, config, target):
        super().__init__(config, target)
        self.results = None  # to store the VARResults instance
        self._complete_config()  # add default values of missing arguments

    def _complete_config(self):
        if "maxlags" not in self.config:
            self.config["maxlags"] = 1
        if "ic" not in self.config:
            self.config["ic"] = None

    def createModel(data):
        # substitute the VAR model to an AR model when only the target variable is in the data
        if data.shape[1] > 1:
            model = VAR(data)
        else:
            if self.config["ic"] is not None:
                lag_order = ar_select_order(data, maxlag=self.config["maxlags"], ic=self.config["ic"])
                ar_lags = lag_order.ar_lags
            else:
                ar_lags = range(1, self.config["maxlags"]+1)
            model = AutoReg(data, lags=ar_lags)
        return model
            
    def fit(self, data):
        self.data = data
        self.model = createModel(data)
        
        if data.shape[1] > 1:
            self.results = self.model.fit(**self.config)
        else:
            self.results = self.model.fit()

    def fittedvalues(self, data=None):
        if data is None:
            fittedvalues = self.results.fittedvalues.dropna()[self.target]
        else:
            # separate data into many one-step intervals)
            maxlags = self.config["maxlags"]
            indexes = data.index[range(maxlags, len(data)-maxlags)]
            datawindows = [data.values[i: i+maxlags] for i in range(len(data)-maxlags-1)]
            totres = []
            for dataitem in datawindows:
                totres.append(self.results.forecast(dataitem, steps = 1))
            totres = np.concat(totres)
            df = pd.DataFrame(totres, index = indexes, columns = data.columns)
            fittedvalues = df[self.target]
        return fittedvalues
    
    def residuals(self, data=None):
        if data is None:  # train residuals
            resid = self.results.resid.dropna()
            if len(resid.shape) == 1:  # result of AR model
                residuals = pd.DataFrame({self.target: resid})
            else: #result of VAR model
                residuals = resid[[self.target]]
        else:  # residuals on provided test data
            residuals = super().residuals(data)
        return residuals

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
        
        elif method == "f-test":
            tested_variable = [x for x in self.data.columns if x not in previous_model.data.columns][0]
            metric = self.results.test_causality(self.target,causing=tested_variable,
                                                 kind="f").pvalue
        
        elif method =="wald-test":
            tested_variable = [x for x in self.data.columns if x not in previous_model.data.columns][0]
            metric = self.results.test_causality(self.target,causing=tested_variable,
                                                 kind="wald").pvalue
                                                                 
        elif method == "by_hand_f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
            
        elif method == "lr-test":
            diff_dof = 0
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    diff_dof+=1
            cstat = -2*(previous_model.llh() - self.llh())
            metric = chi2.sf(cstat,df=diff_dof)
            
        return metric

    def aic(self):
        return self.results.aic
        
    def llh(self):
        return self.results.llf

    def dof(self):
        return self.results.df_resid

    
    def has_too_many_parameters(self):
        nbparams = len(self.results.params)
        nobs = self.results.nobs
        return nobs/nbparams<10
        
    def statistics(self, data=None):
        # handle statistics on both training and test sets
        # the .apply method does not work for distributed lags, so I have to rely on a hack.
        # also, I did not implement llf computation for the test model.
        statistics = super().statistics(data)
        if data is None:
            #add train-specific stats
            statistics["llh"] = self.llh()
            statistics["aic"] = self.aic()
        return statistics



class ARDLModel(LearningModel):
    """
    Adapted from statsmodels.tsa.ardl.ARDL.
     - due to arguments in both instance declaration and fit routine, config must contain:
       - arguments to pass to ARDL constructor
       - arguments to pass to ARDL.fit
     - does not use lag estimation
     - target variable is included in the observed variables
     - uses minus aic of final model as significance

    config (dict):
     - "constructor" (dict): arguments to pass to the ARDL constructor
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.html#statsmodels.tsa.ardl.ARDL
     - "fit" (dict): arguments to pass to the ARDL.fit method
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.fit.html#statsmodels.tsa.ardl.ARDL.fit
    """

    def __init__(self, config, target):
        super().__init__(config, target)
        self.results = None  # to store the VARResults instance

    def fit(self, data):
        self.data = data
        self.model = self.createModel(data)
        self.results = self.model.fit(**self.config["fit"])
        
    def createModel(self, data):
        """
        Creates the model from the data provided.
        Created model is not trained.
        """
        if len(data.columns)>1:
            model = ARDL(endog=data[self.target],
                              exog=data.loc[:, data.columns != self.target],
                              **self.config["constructor"])
        else:
            model = ARDL(endog=data[self.target],
                              exog=None,
                              order=None,
                              **pydash.omit(self.config["constructor"], "order"))
        return model

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
            
        elif method == "f-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.f_test(r_matrix).pvalue
            
        elif method == "by_hand_f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        
        elif method == "wald-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.wald_test(r_matrix, use_f=False, scalar=True).pvalue
        
        elif method == "lr-test":
            diff_dof = 0
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    diff_dof+=1
            cstat = -2*(previous_model.llh() - self.llh())
            metric = chi2.sf(cstat,df=diff_dof)
        
        return metric

    def _pad_test_data_to_create_model(self, data):
        """
        The not-so-nice thing about creating a copy model for test data,
        is that model instanciation checks that the data is large enough to learn.
        Hence, when test data is large enough to be evaluated (timesteps > lags)
        but not enough to be learned (timesteps < lags*variables + 1), it blocks.
        The solution I found is to simply pad the data. The fittedvalue method will select
        the right timestamps at the end.
        This method is handling the padding.
        
        Returns:
             the padded or nonpadded data
        Raises NotEnoughDataError if the data is too small for even 1 prediction.
        """
        if len(data)<=self.config["constructor"]["lags"] or len(data)<=self.config["constructor"]["order"]:
            # estimation is impossible, not enough descriptors for 1 prediction
            raise NotEnoughDataError(len(data), self.config["constructor"]["lags"], self.config["constructor"]["order"])
        
        period = self.config["constructor"]["period"] if "period" in self.config["constructor"] else 1
        period = period+1 if period is not None else 2
        needed_regressors = (len(data.columns) - 1)*self.config["constructor"]["order"] 
        needed_regressors += self.config["constructor"]["lags"] + 4
        needed_regressors *= period
        if needed_regressors - len(data)>0:
            zeros = np.zeros((needed_regressors - len(data), len(data.columns)))
            index = range(len(zeros))
            zeros = pd.DataFrame(zeros, columns=data.columns, index=index)
            data = pd.concat([data, zeros])
            
        return data

    def fittedvalues(self,data=None):
        if data is not None:
            index = data.index  # keep track of original index
            pad_data = self._pad_test_data_to_create_model(data)  # pad just in case test size is small
            pad_data = pad_data.reset_index(drop=True)  # predict works better for rangeindex starting at 0
            model = self.createModel(pad_data)
            # use previous parameters 
            fittedvalues = model.predict(self.results._params, start=0,end=len(data)-1, dynamic=False)
            fittedvalues_nona = fittedvalues.dropna()
            fittedvalues_nona.index = index[fittedvalues_nona.index]
            if len(fittedvalues_nona)==0:
                print(fittedvalues_nona)
                print(fittedvalues)
                print(pad_data)
                print("\n")
            return fittedvalues_nona
        else:
            return self.results.fittedvalues

    
    def residuals(self, data=None):
        if data is not None:
            df = super().residuals(data)
        else:
            residuals = self.results.resid.dropna()
            df = pd.DataFrame({self.target: residuals})
        return df

    def aic(self):
        return self.results.aic
        
    def llh(self):
        return self.results.llf

    def dof(self):
        return self.results.df_resid
    
    def has_too_many_parameters(self):
        nbparams = len(self.results.params)
        nobs = self.results.nobs
        return nobs/nbparams<10
    
    def statistics(self, data=None):
        # handle statistics on both training and test sets
        # the .apply method does not work for distributed lags, so I have to rely on a hack.
        # also, I did not implement llf computation for the test model.
        statistics = super().statistics(data)
        if data is None:
            #add train-specific stats
            statistics["llh"] = self.llh()
            statistics["aic"] = self.aic()
        return statistics
        



class VARMAModel(LearningModel):
    def __init__(self, config, target):
        super().__init__(config, target)
        self.target = target
        self.model = None
        self.results = None  # to store the VARResults instance
        self.config = config

        self.data = None

    def verify_assumption(self):
        pass

    def fit(self, data):
        self.data = data
        # separate initial univariate case from general case with exogenous variables
        if len(data.columns) > 1:
            self.model = VARMAX(endog=data, **self.config["constructor"])
        else:
            p, q = self.config["constructor"]["order"]
            self.model = ARIMA(endog=data[self.target],
                               order=(p, 0, q),
                               **pydash.omit(self.config["constructor"], "order"))
        self.results = self.model.fit(**self.config["fit"])

    def residuals(self):
        resid = self.results.resid
        if len(resid.shape) == 1:  # result of ARMA model
            df = pd.DataFrame({self.target: resid})
            # df.loc[len(df)] = [np.nan]
            return df
        return resid[[self.target]]

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model.
        
        Note: Usually, df_model = nobs-df_resid. So it should not matter which one we use.
        VARMA is tricky so this has to be verified.
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
        elif method == "f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        elif method == "llhr-test":
            statistic = 2*(self.llh() - previous_model.llh())
            pvalue = chi2.sf(statistic, previous_model.dof() - self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        return metric

    def aic(self):
        return self.results.aic

    def dof(self):
        return self.results.df_resid

    def sse(self):
        return pd.Series.sum(self.residuals()[self.target]**2)


class SARIMAXModel(LearningModel):
    def __init__(self, config, target):
        super().__init__(config, target)
        self.target = target
        self.model = None
        self.results = None  # to store the VARResults instance
        self.config = config

        self.data = None

    def fit(self, data):
        self.data = data
        if len(data.columns) > 1:
            self.model = SARIMAX(endog=data[self.target],
                                 exog=data.loc[:, data.columns != self.target],
                                 **self.config["constructor"])
        else:
            self.model = SARIMAX(endog=data[self.target],
                                 exog=None,
                                 **self.config["constructor"])
        self.results = self.model.fit(**self.config["fit"])

    def residuals(self):
        resid = self.results.resid
        df = pd.DataFrame({self.target: resid})
        return df

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
        elif method == "f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        return metric

    def aic(self):
        return self.results.aic

    def sse(self):
        return pd.Series.sum(self.residuals()[self.target]**2)

    def dof(self):
        # there is no explicit way to have the degree of freedom of the model.
        # the same method as the normality test, heteroscedasticity test, and serial test is applied.
        d = np.maximum(self.results.loglikelihood_burn, self.results.nobs_diffuse)
        nobs_effective = self.results.nobs - d
        return nobs_effective

##############
#
#   LSTM models trials
#
##############

class LSTMModel(LearningModel):
    """
    Class that uses the DeepAR implementation.
    I predict the value of target_t, with the information of target_t-1 and covariates_t.
    Currently the loss is seq2seq, meaning that all intermediary terms count, not just the final predicted target.
    It uses a probabilistic implementation, meaning that log-likelyhood can be computed.
    Currently, does not add any structure to the input time series.
    Currently, the predicted target is expected to follow a conditional gaussian law.
    Uses one-step forecasting for residual computation.
    """
    def __init__(self, config, target):
        super().__init__(config, target)
        self.target = target
        self.model = None
        self.config = config

        self.data = None
        self.params = None
        self.train_loader = None
        self._resids = None
        self._llh = None
        self._training_last_loss = None

    def verify_assumption(self):
        pass

    def _create_params(self):
        params = {
            "cov_dim": len(self.data.columns)-1,  # number of instantaneous covariates
            "lstm_hidden_dim": self.config["lstm_hidden_dim"],
            "lstm_layers": self.config["lstm_layers"],
            "lstm_dropout": self.config["lstm_dropout"],
            "device": device("cuda" if torch.cuda.is_available() else "cpu"),
            "sample_times": 1,  # only generate one instance of each predicted variable
            "predict_steps": 1,  # only predict the next time step
            "predict_start": self.config["maxlags"],  # a bit of a wrong design in our case:
                                                      # the start time we predict values of target from
            "train_window": self.config["maxlags"]+1,  # +1 since we include lag 0
            "batch_size": self.config["batch_size"],
            "sampling": self.config["sampling"],  # whether to produce residuals by sampling or distribution mean
            "loss_predict_only": self.config["loss_predict_only"]  # True means that only the last prediction is used
        }
        return params

    def fit(self, data):
        self.data = data
        self.params = self._create_params()

        self.model = deepAR.Net(self.params).to(self.params["device"])
        self.train_loader = deepAR.single_mts_to_dataloader(data, self.target, self.config["maxlags"]+1,
                                                            self.config["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self._training_last_loss = deepAR.train(self.model, optimizer, deepAR.llh_loss_fn,
                                                self.config["epochs"], self.train_loader, self.params)

    def residuals(self):
        if self._resids is None:
            test_loader = deepAR.single_mts_to_dataloader(self.data, self.target, self.config["maxlags"]+1,
                                                          1, shuffle=False)
            resid = deepAR.compute_residuals(self.model, test_loader, self.params)

            new_index = self.data.index[range(self.config["maxlags"], len(self.data))]  # only select valid index
            resid_df = pd.DataFrame(resid, index=new_index, columns=[self.target])
            self._resids = resid_df
        return self._resids

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the metric the better the new model
        """
        metric = None
        if method == "llh":
            previous_model_significance = - previous_model.llh()
            current_model_significance = - self.llh()
            metric = current_model_significance - previous_model_significance
        if method == "sse":
            previous_model_significance = previous_model.sse()
            current_model_significance = self.sse()
            metric = current_model_significance - previous_model_significance
        return metric

    def llh(self):
        """Uses the output distribution to compute the average loglikelyhood of the data."""
        if self._llh is None:
            test_loader = deepAR.single_mts_to_dataloader(self.data, self.target, self.config["maxlags"]+1,
                                                          self.config["batch_size"], shuffle=False)
            llh = deepAR.compute_log_likelyhood(self.model, test_loader, self.params)
            self._llh = llh / len(self.data - self.config["maxlags"])
        return self._llh

    def sse(self):
        return pd.Series.sum(self.residuals()[self.target]**2)

    def last_train_loss(self):
        return self._training_last_loss


class LSTMModelKfold(LSTMModel):
    """
    Modification of the LSTM DeepAR model to separate train and test according to the Forward Chaining CV.
    If there are k folds, only the last k/k+1 fraction of the data is used to compute residuals and metrics.
    The only additional config argument is "numberfolds".
    """

    
    def fit(self, data):
        self.data = data
        self.params = self._create_params()

        # get the dataloaders for training and the TSDatasets for evaluating
        self.train_loader, self.start_test = deepAR.single_mts_to_k_dataloaders(data,
                                                                                self.target, self.config["maxlags"]+1,
                                                                                self.config["batch_size"],
                                                                                self.config["numberfolds"])

        # build the k models that will be trained
        self.model = [deepAR.Net(self.params).to(self.params["device"]) for k in range(self.config["numberfolds"])]

        self._training_last_loss = 0
        for k in range(self.config["numberfolds"]):
            optimizer = torch.optim.Adam(self.model[k].parameters(), lr=self.config["learning_rate"])
            self._training_last_loss += deepAR.train(self.model[k], optimizer, deepAR.llh_loss_fn,
                                                     self.config["epochs"], self.train_loader[k][0], self.params)
        self._training_last_loss /= self.config["numberfolds"]  # loss is the mean over samples

    def residuals(self):
        if self._resids is None:
            residlist = []
            for k in range(self.config["numberfolds"]):
                test_loader = deepAR.DataLoader(self.train_loader[k][1], batch_size=1, shuffle=False)
                resid = deepAR.compute_residuals(self.model[k], test_loader, self.params)
                residlist.extend(resid[:, 0])
            # create index corresponding to the residuals
            new_index = range(self.start_test, len(self.data))
            new_index = self.data.index[new_index]
            # create result dataset
            resid_df = pd.DataFrame(residlist, index=new_index, columns=[self.target])
            self._resids = resid_df
        return self._resids

    def llh(self):
        """Uses the output distribution to compute the average loglikelyhood of the data."""
        if self._llh is None:
            llh = 0
            for k in range(self.config["numberfolds"]):
                test_loader = deepAR.DataLoader(self.train_loader[k][1], batch_size=256, shuffle=False)
                llh += deepAR.compute_log_likelyhood(self.model[k], test_loader, self.params)
            number_windows = len(self.data)-self.config["maxlags"]+1+1
            self._llh = llh / number_windows
        return self._llh
        
        
class LSTMModelTrainVal(LSTMModel):
    """
    Modification of the LSTM DeepAR model to separate train and val according to a single forward split.
    Configurable split fraction.
    Bootstrapping with replacement used on the validation set for stopping criterion
        Since the datasplit is the same for all models, we can use the same permutation, and a wilcoxon paired
        rank test for the significativity of the criterion.
    Only "sse" currently available as stopping method.
    """
        
    def _separate_train_val(self, data):
        number_effective_timesteps = len(data) - 2*(self.config["maxlags"])
        number_train_instances = number_effective_timesteps * (1-self.config["train_val_split"])
        index_split = int(number_train_instances + self.config["maxlags"]+1)
        train_data = data.iloc[:index_split]
        val_data = data.iloc[index_split:]
        return train_data, val_data
        
    def fit(self, data):
        self.data = data
        self.params = self._create_params()

        # get the dataloader for training and validation
        train_data, val_data = self._separate_train_val(data)
        self.train_loader = deepAR.single_mts_to_dataloader(train_data, self.target, self.config["maxlags"]+1,
                                                            self.config["batch_size"], shuffle=True)
        self.val_loader = deepAR.single_mts_to_dataloader(val_data, self.target, self.config["maxlags"]+1,
                                                            self.config["batch_size"], shuffle=False)

        # build the models that will be trained
        self.model = deepAR.Net(self.params).to(self.params["device"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self._loss_training, self._metrics_training = deepAR.train(self.model, optimizer, deepAR.llh_loss_fn,
                                                      self.config["epochs"], self.train_loader, self.params,
                                                      self.val_loader)
        self._training_last_loss = self._loss_training[-1]
                                                     
    def llh(self):
        """Uses the output distribution to compute the average loglikelyhood of the data."""
        if self._llh is None:
            llh = deepAR.compute_log_likelyhood(self.model, self.val_loader, self.params)
            self._llh = llh
        return self._llh

    def residuals(self):
        if self._resids is None:
            resid = deepAR.compute_residuals(self.model, self.val_loader, self.params)
            # create index corresponding to the residuals
            _, val_data = self._separate_train_val(self.data)
            new_index = val_data.index[self.config["maxlags"]:]
            # create result dataset
            resid_df = pd.DataFrame(resid, index=new_index, columns=[self.target])
            self._resids = resid_df
        return self._resids


    def _create_resampling_counts(self):
        nbbootstraps = self.config['bootstrap_number']
        nbelements = len(self.residuals())
        sums = np.zeros((nbbootstraps, nbelements), dtype=int)
        for i in range(nbbootstraps):
            for k in range(nbelements):
                sums[i,np.random.randint(0, nbelements-1)] += 1
        return sums
            
    def sse_resampled(self, resamplings):
        sselist = []
        squared_residuals = self.residuals()[self.target]**2
        for sample_count in resamplings:
            sse = pd.Series.sum(squared_residuals * sample_count)
            sselist.append(sse)
        return sselist

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the metric the better the new model
        """
        metric = None
        if method == "sse-bootstrap":
            resamplings = self._create_resampling_counts()
            previous_model_sse = previous_model.sse_resampled(resamplings)
            current_model_sse = self.sse_resampled(resamplings)
            _, metric = wilcoxon(previous_model_sse, current_model_sse, alternative="less")
        return metric

    def statistics(self):
        statistics = dict()
        statistics["sse_val"] = pd.Series.sum(self.residuals()[self.target]**2)
        statistics["sse_train"] = np.sum(deepAR.compute_residuals(self.model, self.train_loader, self.params)**2)
        statistics["llh_val"] = self.llh()
        statistics["llh_train"] = deepAR.compute_log_likelyhood(self.model, self.train_loader, self.params)
        # to compare between train and val, we need the average over samples so dataset size do not matter
        statistics["rmse_val"] = np.sqrt(statistics["sse_val"] / len(self.residuals()))
        statistics["rmse_train"] = np.sqrt(statistics["sse_val"] / len(self.train_loader.dataset))
        statistics["mllh_val"] = self.llh() / len(self.val_loader.dataset)
        statistics["mllh_train"] = statistics["llh_train"] / len(self.train_loader.dataset)
        statistics["final_training_loss"] = self._training_last_loss
        statistics["training_loss"] = self._loss_training
        statistics["metrics_during_train"] = self._metrics_training
        return statistics

