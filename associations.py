from scipy.stats import pearsonr, spearmanr, beta, rankdata
from scipy.special import stdtr
import pingouin

#from mass_ts import mass2

import numpy as np
import pandas as pd

##
#
#   Association classes
#
##

class Association:
    def __init__(self, config):
        self.config = config

    def association(self, residuals_df, variable_df):
        pass


class PearsonCorrelation(Association):
    """
    Data assumption:
     - dataframe is sorted by timestamp increasing
     - timestamps are equidistant
     - data does not have missing values
     - can currently only process single-sample data.
    
    config:
     - return_type (str):
       - p-value: the returned association is minus the p-value
       - coefficient: the returned association is a correlation coefficient
     - lags (int): the maximal lag of the variable to use.
        if set to 0, only the immediate correlation is computed.
        if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
     - selection_rule (str): the rule to use to aggregate the lags
       - max: use maximal correlation/ min p-value (max norm on correlation function / best p-value)
       - average: use average correlation (integral on correlation function)
    """
    def _select_rows_with_lag(self, residuals_df, variable_df, lag):
        """
        This function selects and align valid rows of the residuals and the evaluated variable.
        It uses indexes to compare the two dataframe, and use integer location to compute the lag.
        This part specifically needs observations to be sorted and equidistants in time.
        """
        # select valid residuals (exclude nan)
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        # select the right lag and corresponding rows in both dataframe
        residuals_indexes = set(residuals_df.index)
        variable_ilocs = [i for i in range(variable_df.shape[0]) if variable_df.index[i] in residuals_indexes]
        selected_lagged_ilocs = [i-lag for i in variable_ilocs if i-lag >= 0]
        selected_residual_indexes = [variable_df.index[i+lag] for i in selected_lagged_ilocs]
        res = residuals_df.loc[selected_residual_indexes].values
        var = variable_df.iloc[selected_lagged_ilocs].values
        return res.reshape((-1,)), var.reshape((-1,))
    
    def association(self, residuals_df, variable_df):
        measures = np.zeros((self.config["lags"]+1,))
        
        for lag in range(self.config["lags"]+1):
            res, var = self._select_rows_with_lag(residuals_df, variable_df, lag)
            r, p = pearsonr(res, var)
            
            if self.config["return_type"] == "p-value":
                measures[lag] = -p
            elif self.config["return_type"] == "coefficient":
                measures[lag] = abs(r)
                
        if self.config["selection_rule"] == "max":
            return np.max(measures)
        elif self.config["selection_rule"] == "average":
            return np.mean(measures)


class SpearmanCorrelation(Association):
    """
    Data assumption:
     - dataframe is sorted by timestamp increasing
     - timestamps are equidistants
     - data can have missing values, they will be ignored.
     - can currently only process single-sample data.
    
    config:
     - return_type (str):
       - p-value: the returned association is minus the p-value
       - coefficient: the returned association is a correlation coefficient
     - lags (int): the maximal lag of the variable to use.
        if set to 0, only the immediate correlation is computed.
        if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
     - selection_rule: the rule to use to aggregate the lags
       - max: use maximal correlation/ min p-value (max norm on correlation function / best p-value)
       - average: use average correlation (integral on correlation function)
    """
    def _select_rows_with_lag(self, residuals_df, variable_df, lag):
        """
        This function selects and align valid rows of the residuals and the evaluated variable.
        It uses indexes to compare the two dataframe, and use integer location to compute the lag.
        This part specifically needs observations to be sorted and equidistants in time.
        """
        # select valid residuals (exclude nan)
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        # select the right lag and corresponding rows in both dataframe
        residuals_indexes = set(residuals_df.index)
        variable_ilocs = [i for i in range(variable_df.shape[0]) if variable_df.index[i] in residuals_indexes]
        selected_lagged_ilocs = [i-lag for i in variable_ilocs if i-lag >= 0]
        selected_residual_indexes = [variable_df.index[i+lag] for i in selected_lagged_ilocs]
        res = residuals_df.loc[selected_residual_indexes].values
        var = variable_df.iloc[selected_lagged_ilocs].values
        return res.reshape((-1,)), var.reshape((-1,))
        
    def association(self, residuals_df, variable_df):
        measures = np.zeros((self.config["lags"]+1,))
        
        for lag in range(self.config["lags"]+1):
            res, var = self._select_rows_with_lag(residuals_df, variable_df, lag)
            r, p = spearmanr(res, var, nan_policy="omit")
            
            if self.config["return_type"] == "p-value":
                measures[lag] = -p
            elif self.config["return_type"] == "coefficient":
                measures[lag] = abs(r)
                
        if self.config["selection_rule"] == "max":
            return np.max(measures)
        elif self.config["selection_rule"] == "average":
            return np.mean(measures)



class UsingMASS(Association):
    """
    Computes for each lag up to <lags> of the given variable, its <return_type> with the residuals.
    The result is then aggregated into a single score using <selection_rule>.

        Prefered use case:
         - many lags have to be computed

        Data assumption:
         - dataframe is sorted by timestamp increasing
         - timestamps are equidistants
         - data has no missing value
         - can currently only process single-sample data.
         - the last <lags> values of the tested variable will be excluded.
         - the first values of the tested variable will be excluded, depending on the LearningModel lag, to correspond
           to residuals.

        config:
         - return_type (str):
           - distance: the computed association is minus the normalized euclidean distance
           - correlation: the computed association is the pearson correlation
           - p-value: the computed association is the p-value of the pearson correlation
         - lags (int): the maximal lag of the variable to use.
            if set to 0, only the immediate correlation is computed.
            if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
         - selection_rule: the rule to use to aggregate the lags
           - max: use maximal correlation / min distance / minimal p-value
           - average: use average correlation / average distance / average p-value
        """

    def _select_correct_rows(self, residuals_df, variable_df):
        # need to adjust the time axis of both dataframes
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        residuals_indexes = set(residuals_df.index)
        variable_ilocs = [i for i in range(variable_df.shape[0]) if (variable_df.index[i] in residuals_indexes)]
        variable_ilocs = variable_ilocs[:-self.config["lags"]]
        residual = residuals_df.values.reshape((-1,))
        variable = variable_df.iloc[variable_ilocs].values.reshape((-1,))
        return residual, variable

    def association(self, residuals_df, variable_df):
        residual, variable = self._select_correct_rows(residuals_df, variable_df)

        coefficients = mass2(residual, variable)  # compute distances

        if self.config["return_type"] == "correlation":
            coefficients = 1 - np.absolute(coefficients)**2 / (2 * len(variable))
        elif self.config["return_type"] == "p-value":
            coefficients = 1 - np.absolute(coefficients)**2 / (2 * len(variable))
            # next 3 lines taken from scipy.stats.pearsonr
            ab = len(variable)/2 - 1
            beta_distribution = beta(ab, ab, loc=-1, scale=2)
            coefficients = - 2 * beta_distribution.sf(np.abs(coefficients))
        else:
            coefficients = - coefficients

        if self.config["selection_rule"] == "max":
            return np.max(coefficients)
        elif self.config["selection_rule"] == "average":
            return np.mean(coefficients)
            
########################################################################
#
#
#
########################################################################



def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
def moving_average(a, window=3):
    return np.mean(rolling_window(a, window), -1)
def moving_std(a, window=3):
    return np.std(rolling_window(a, window), -1)
def mass2_modified(ts, query):
    #adapted from the mass-ts module, to allow for 2D ts input.
    #ts is an array of form (time, variables) and query of form (time,).
    ts, query = np.array(ts), np.array(query)
    n = len(ts)
    v = ts.shape[-1]
    m = len(query)
    x = ts.T
    y = query

    meany = np.mean(y)
    sigmay = np.std(y)

    meanx = moving_average(x, m)
    meanx = np.append(np.ones([v, m - 1]), meanx, axis=-1)

    sigmax = moving_std(x, m)
    sigmax = np.append(np.zeros([v, m - 1]), sigmax, axis=-1)

    y = np.append(np.flip(y), np.zeros([1, n - m]))

    X = np.fft.fft(x,axis=-1)
    Y = np.fft.fft(y,axis=-1)
    Z = X * Y
    z = np.fft.ifft(Z,axis=-1)

    dist = 2 * (m - (z[:,m - 1:n] - m * meanx[:,m - 1:n] * meany) /
                (sigmax[:,m - 1:n] * sigmay))

    correlation = 1 - np.absolute(dist) / (2 * m)

    return correlation


class PearsonMultivariate(Association):
    """
    Computes for each lag up to <lags> of the given variables, its <return_type> with the residuals.
    The result is then aggregated into a single score using <selection_rule>.

        Prefered use case:
         - many lags have to be computed

        Data assumption:
         - dataframe is sorted by timestamp increasing
         - timestamps are equidistants
         - data has no missing value
         - can currently only process single-sample data.
         - the first <lags> non-na values of the residuals will be excluded.
         - the first values of the tested variable are excluded, depending on the LearningModel lag, to correspond
           to residuals.

        config:
         - return_type (str):
           - correlation: the computed association is the pearson correlation
           - p-value: the computed association is the p-value of the pearson correlation
         - lags (int): the maximal lag of the variable to use.
            if set to 0, only the immediate correlation is computed.
            if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
         - selection_rule: the rule to use to aggregate the lags
           - max: use maximal correlation / minimal p-value
           - average: use average correlation / average p-value
        """

    def _select_correct_rows(self, residuals_df, variables_df):
        # remove nans
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        residuals_indexes = set(residuals_df.index)
        #adjust variable timestamps to residuals since learning process lags will have reduced the length of the series
        variables_ilocs = [i for i in range(variables_df.shape[0]) if (variables_df.index[i] in residuals_indexes)]
        #remove the first <lags> elements of the residuals for mass2_modified computation.
        residuals_ilocs = [i for i in range(residuals_df.shape[0])]
        residuals_ilocs = residuals_ilocs[self.config["lags"]:]
        
        residuals = residuals_df.iloc[residuals_ilocs].values.reshape((-1,))
        variables = variables_df.iloc[variables_ilocs].values
        return residuals, variables

    def association(self, residuals_df, variables_df):
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        coefficients = mass2_modified(variables, residuals)  # compute correlations

        if self.config["return_type"] == "p-value":
            # next 3 lines taken from scipy.stats.pearsonr
            ab = len(residuals)/2 - 1  # len(residuals) is the total sample size over which correlation is computed
            beta_distribution = beta(ab, ab, loc=-1, scale=2)
            coefficients = - 2 * beta_distribution.sf(np.abs(coefficients))

        if self.config["selection_rule"] == "max":
            return np.max(coefficients, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(coefficients, axis=-1)


class SpearmanMultivariate(PearsonMultivariate):
    """
    Computes for each lag up to <lags> of the given variables, its <return_type> with the residuals.
    The result is then aggregated into a single score using <selection_rule>.

        Prefered use case:
         - many lags have to be computed

        Data assumption:
         - dataframe is sorted by timestamp increasing
         - timestamps are equidistants
         - data has no missing value
         - can currently only process single-sample data.
         - the first <lags> non-na values of the residuals will be excluded.
         - the first values of the tested variable are excluded, depending on the LearningModel lag, to correspond
           to residuals.

        config:
         - return_type (str):
           - correlation: the computed association is the pearson correlation
           - p-value: the computed association is the p-value of the pearson correlation
         - lags (int): the maximal lag of the variable to use.
            if set to 0, only the immediate correlation is computed.
            if > 0, the lag of maximal correlation / minimal p-value amongst the lags is selected.
         - selection_rule: the rule to use to aggregate the lags
           - max: use maximal correlation / minimal p-value
           - average: use average correlation / average p-value
        """
    def _compute_ranks(self,residuals,variables):
        rr = rankdata(residuals)
        rv = rankdata(variables,axis=0)
        return rr,rv

    def association(self, residuals_df, variables_df):
        #align mts
        residuals, variables = self._select_correct_rows(residuals_df, variables_df)

        #spearman computation
        residuals, variables = self._compute_ranks(residuals,variables)
        coefficients = mass2_modified(variables, residuals)

        #pvalues
        if self.config["return_type"] == "p-value":
            # next lines taken from scipy.stats
            dof = len(residuals) - 2
            # test statistic
            coefficients = coefficients * np.sqrt((dof/((coefficients+1.0)*(1.0-coefficients))).clip(0))
            # comparision with student t
            coefficients = stdtr(dof, -np.abs(coefficients))*2

        if self.config["selection_rule"] == "max":
            return np.max(coefficients, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(coefficients, axis=-1)


class LinearPartialCorrelation():
    def __init__(self, config):
        self.config = config
        self._check_config()
        
    def _check_config(self):
        assert "method" in self.config
        assert "lags" in self.config
        assert "selection_rule" in self.config
    
    def _prepare_data(self, condition_df, residuals_df, candidate_df):
        """
        Pingouin partial_corr asks for the data in form of a dataframe.
        This function formats the lags of condition, candidate and joins them with residuals.
        """
        # remove nans eventually occuring in residuals
        residuals_df = residuals_df[~residuals_df.isnull().any(axis=1)]
        
        # add lags of the condition variable
        col_name = condition_df.columns[0]
        condition_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            condition_cols[col_name+"lag -"+str(lag)] = condition_df[col_name].shift(lag)
        condition_cols = condition_cols.iloc[self.config["lags"]:]
        
        # add lags of the tested variable
        col_name = candidate_df.columns[0]
        candidate_cols = pd.DataFrame()
        for lag in range(1,self.config["lags"]+1):
            candidate_cols[col_name+"lag -"+str(lag)] = candidate_df[col_name].shift(lag)
        candidate_cols = candidate_cols.iloc[self.config["lags"]:]
        
        # create new index
        new_index = residuals_df.index.intersection(condition_cols.index)
        residuals_df = residuals_df.loc[new_index]
        candidate_cols = candidate_cols.loc[new_index]
        condition_cols = condition_cols.loc[new_index]
        
        # concatenate
        df = pd.concat([residuals_df, candidate_cols, condition_cols],axis=1)
        cond_names = condition_cols.columns
        cand_names = candidate_cols.columns
        return df, cond_names, cand_names
    
    def partial_corr(self, residuals_df, candidate_df, condition_df):
        """
        Compute the partial correlation of residuals_df with each lag of candidate_df by taking condition_df variable into account.
        """
        data, cond_names, cand_names = self._prepare_data(condition_df, residuals_df, candidate_df)
        resid_name = residuals_df.columns[0]
        
        pvals = []
        for cand_name in cand_names:
            res = pingouin.partial_corr(data, x=resid_name, y=cand_name, covar=list(cond_names), method=self.config["method"])
            pvals.append(res["p-val"].values[0])
        
        if self.config["selection_rule"] == "min":
            return np.min(pvals, axis=-1)
        elif self.config["selection_rule"] == "average":
            return np.mean(pvals, axis=-1)
            
            
