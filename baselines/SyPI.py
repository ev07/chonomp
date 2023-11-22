from collections import defaultdict
import scipy
from typing import Tuple
import pingouin
#from partialcorrel import partialcorrel
#from calculate_metrics import calculate_false_negatives, calculate_false_positives
import numpy as np
import pandas as pd
import copy


from numpy import linalg as LA
import glmnet_python
from glmnet import glmnet

from rpy2.robjects.packages import importr, data
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()
from rpy2.robjects import default_converter, conversion
glmnet = importr("glmnet")


def lasso_granger(series, P):
    """Lasso Granger
    A. Arnold, Y. Liu, and N. Abe. Temporal causal modeling with graphical granger methods. In KDD, 200
    :param series: (N,T) matrix
    :param P: length of the lag
    :param alpha: value of the penalization parameter in Lasso
    :return:
        array NxT with arr[0,i] the ith lag coefficient
    """
    N, T = np.shape(series)
    Am = np.zeros((T - P, P * N))
    bm = np.zeros((T - P, 1))
    for i in range(P, T):
        bm[i - P] = series[0, i]
        Am[i - P, :] = np.fliplr(series[:, i - P:i]).flatten()

    Am2 = pd.DataFrame(Am)
    bm2 = pd.Series(bm[:,0])
    with conversion.localconverter(default_converter + pandas2ri.converter):
        Am2 = ro.conversion.get_conversion().py2rpy(Am2)
        bm2 = ro.conversion.get_conversion().py2rpy(bm2)
        
        # Lasso using GLMnet
        fit = glmnet.glmnet(x=Am2, y=bm2, family='gaussian', alpha=1)
        #vals2 = ro.r.predict(fit,type="coefficients")
        
        vals2 = fit['beta']  # array of coefficient
        vals2 = ro.r["as.matrix"](vals2)
    
    # Using the AIC to select lambda
    tLL = - (1-fit["dev.ratio"])*fit["nulldev"]
    k = fit["df"]
    aic = -tLL + 2*k
    solution_index = np.argmin(aic)
    vals3 = vals2[:,solution_index]

    # Reformatting the results into (N,P) matrix
    n1Coeff = np.zeros((N, P))
    for i in range(N):
        n1Coeff[i, :] = vals3[i * P:(i + 1) * P].reshape(P)

    return n1Coeff

def normalize_input(X: np.ndarray, normalized_data: bool, normalization: str, n_time_series: int) -> np.ndarray:
    """
    Normalize data X or not according to normalization type

    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :param normalized_data: True or False
    :param normalization: 'variance' or 'minmax'
    :param n_time_series: int number of time series
    :return: Xnorm normalized time series if normalized_data==True
    """
    if normalized_data:
        if normalization == 'variance':
            # variance normalization
            Xnorm = np.full(np.shape(X), np.nan)
            for i_var in range(n_time_series):
                if np.count_nonzero(X[i_var, :]) == 0:
                    Xnorm[i_var, :] = X[i_var, :]
                else:
                    Xnorm[i_var, :] = (X[i_var, :] - np.mean(X[i_var, :])) / np.std(X[i_var, :])
        if normalization == 'minmax':
            # min max normalization
            Xnorm = np.full(np.shape(X), np.nan)
            for i_var in range(n_time_series):
                if np.count_nonzero(X[i_var, :]) == 0:
                    Xnorm[i_var, :] = X[i_var, :]
                else:
                    Xnorm[i_var, :] = (X[i_var, :] - np.min(X[i_var, :])) / (np.max(X[i_var, :]) - np.min(X[i_var, :]))
    else:
        Xnorm = copy.deepcopy(X)
    return Xnorm


def calculate_w_wmax(X: np.ndarray, threshold: float, lambda_reg: float, regression_algorithm: str, normalized_data: bool, normalization: str, order) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the lag between each time series and the target time series

    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :param threshold: float threshold for the coefficients of lasso regression
    :param lambda_reg: float lambda regularizer for the lasso regression
    :param regression_algorithm: string, until now only scipy_lassoalgo implemented.
    :param normalized_data: True if you want to normalize the data, False if no normalization is requested
    :param normalization: string, type of normalization 'variance' or 'minmax'
    :return: w_max the maximum lag among all lags, w 1D np.ndarray where each value corresponds to the lag of that time
    series with the target. If there is no dependency between a time series i and the target then w[i] = nan
    """

    n_time_series, n_time_steps = np.shape(X)
    # normalize data or not according to what user requests
    Xnorm = normalize_input(X, normalized_data, normalization, n_time_series)
    n_vars = n_time_series - 1  # candidate time series, all apart from the last one which is the target
    w = np.full(n_vars, np.nan) # initialize lag vector
    if regression_algorithm == 'scipy_lassoalgo':
        # apply Granger Causal Inference
        stats = defaultdict(list)
        for step in range(n_time_steps):
            # as many variables as in n_vars
            for iVar in range(n_time_series):
                nameofvar = 'var' + str(iVar)
                stats[nameofvar].append(Xnorm[iVar, step])
        data = pd.DataFrame().from_dict(stats)
        for i_var in range(n_vars):
            data_pair = X[[n_vars, i_var],:]  #target must be first for the implementation of lasso here.
            adjacency_matrix = lasso_granger(data_pair, order) 
            maxlag_iloc = np.argmax(np.abs(adjacency_matrix[1]))
            if abs(adjacency_matrix[1, maxlag_iloc]) > threshold: 
                w[i_var] = maxlag_iloc

    return np.nanmax(w), w


def extract_Xi_t_and_Y_t_wi_for_each_t(observations: np.ndarray, X: np.ndarray, w: np.ndarray, i_candidate: int, n_vars: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From the original time series extract the fourplets X^i_{t-1} (we use xo as var), X^i_t (we use x1 as var),
    Y_{t+wi} (we use Y_wi as var) and Y_{t+wi-1} (we use Y_wi_minus_1 as var) at each point t.

    :param observations: the indices of time points so that it includes the oldest cause of target
    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :param w: 1D numpy array with the lag wi of each time series with the target
    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :param n_vars: int number of candidate time series without the target
    :return: x1, xo, Y_wi, Y_wi_minus_1
    """

    # initialize vars
    x1 = np.full((n_vars, observations.size), np.nan) # X^i_t form the paper
    xo = np.full((n_vars, observations.size), np.nan) # X^i_{t-1} from the paper

    x1[i_candidate, :] = X[i_candidate, observations]  # t
    xo[i_candidate, :] = X[i_candidate, observations - 1]  # t - 1

    # extract the corresponding node of the target time series for each observation
    if not np.isnan(w[i_candidate]):
        Y_wi_minus_1 = X[-1, observations + int(w[i_candidate]) - 1]  # t + wx Y_{t+wi-1} from the paper
        Y_wi = X[-1, observations + int(w[i_candidate])]  # t + wx + 1 Y_{t+wi} from the paper
    else:
        Y_wi_minus_1 = np.full((1, observations.size), np.nan)
        Y_wi = np.full((1, observations.size), np.nan)

    return x1, xo, Y_wi, Y_wi_minus_1


def calculate_set_Si(xo: np.ndarray, x1: np.ndarray, X: np.ndarray, w: np.ndarray, observations: np.ndarray, n_vars: int, i_candidate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract form the time series the node of each time series j != i_candidate that should be added in the S^i conditioning set at each t and put it at x1[j] (we name this x1)

    :param xo: 2D np.ndarray (n_vars) x (n_time_steps): Initially contains nan apart from the xo[i_candidate] that contains X^i_{t-1} at each t
    :param x1: 2D np.ndarray (n_vars) x (n_time_steps): Initially contains nan apart from the x1[i_candidate] that contains X^i_{t} at each t
    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :param w: 1D numpy array with the lag wi of each time series with the target
    :param observations: the indices of time points so that it includes the oldest cause of target
    :param n_vars: int number of candidate time series without the target
    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :return: x1, xo
    """
    # extract the nodes of the other time series j != i (not the one which is currently under examination)
    # that enter the node of Y_{t+wi-1} to build the set S^i from the paper
    other_vars = np.setdiff1d(range(n_vars), i_candidate)
    for i_var in other_vars:
        if not np.isnan(w[i_var]) and not np.isnan(w[i_candidate]):
            x1[i_var, :] = X[i_var, observations + int(w[i_candidate]) - int(
                w[i_var]) - 1]  # t + wx - wyi - 1 ( all that enter Y_{t+wi-1}
            # xo for the other_variables will not be used. Only for the i_candidate is required for the tests.
            xo[i_var, :] = X[i_var, observations + int(w[i_candidate]) - int(w[i_var]) - 2]  # t + wx - wyi - 2
    return x1, xo


def calculate_conditioning_set_and_vars(i_candidate: int, X: np.ndarray, n_time_steps: int, w: np.ndarray, w_max: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From the original time series extract the tuplets consisting of the candidate time series under examination X^i
    at each point t, the target that corresponds to Y_{t+wi} and the conditioning set for these two variables which
     corresponds to the X^j time series at t+wi-wj-1 and the target node at Y_{t+wi-1}. Do this for all t to extract
      the tuplets.

    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :param n_time_steps: int number of time steps in X
    :param w: 1D numpy array with the lag wi of each time series with the target
    :param w_max: float the maximum non nan value in w
    :param n_vars: int number of candidate time series without the target
    :return: xo, x1, Y_wi_minus_1, Y_wi
    """
    # extract form the time series the tuplets of nodes needed for each observation
    observations = np.arange(int(w_max), n_time_steps - int(w_max) - 1)

    x1, xo, Y_wi, Y_wi_minus_1 = extract_Xi_t_and_Y_t_wi_for_each_t(observations, X, w, i_candidate, n_vars)

    # extract the nodes of the other time series j != i (not the one which is currently under examination)
    x1, xo = calculate_set_Si(xo, x1, X, w, observations, n_vars, i_candidate)

    return xo, x1, Y_wi_minus_1, Y_wi


def calculate_pMRplain(n_vars: int, x1: np.ndarray, Y_wi: np.ndarray, xo: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Calculate the p value of the stimple correlation between the nodes in x1 and the target
    Note: Based on our assumptions this is a redundant test. I just added it for sanity check

    :param n_vars: int number of candidate time series without the target
    :param x1: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -1 step (these are the ones that belong to the S^i
     set of the candidate at this observation
    :param Y_wi: 1D np.ndarray containing at each point the node Y_{t+w_icandidate} from the target time series
    :param xo: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t-1} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -2 step
    :param w: 1D numpy array with the lag wi of each time series with the target
    :return: p_MRplain
    """
    p_MRplain = np.full((n_vars,), np.nan)
    for i in range(n_vars):
        if not np.isnan(w[i]) and not np.isnan(x1[i, :]).any() and not np.isnan(xo[i, :]).any():
            if np.array_equal(x1[i, :], x1[i, :].astype(bool)):
                rho_MRplain, p_MRplain[i] = scipy.stats.pointbiserialr(x1[i, :], Y_wi)
            else:
                rho_MRplain, p_MRplain[i] = scipy.stats.pearsonr(x1[i, :], Y_wi)
        else:
            p_MRplain[i] = np.nan
    return p_MRplain


def calculate_pMR(x1: np.ndarray, Y_wi: np.ndarray, Y_wi_minus_1: np.ndarray, i_candidate: int,
                  other_observed_vars: np.ndarray, n_iidsamples: int, p_MRplain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the p value of the conditional dependence X^i_t _|/|_ Y_{t+wi} | {S^i, Y_{t+wi-1}}
    Condition 1 on paper

    :param x1: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -1 step (these are the ones that belong to the S^i
     set of the candidate at this observation
    :param Y_wi: 1D np.ndarray containing at each point the node Y_{t+w_icandidate} from the target time series
    :param Y_wi_minus_1: 1D np.ndarray containing at each point the node Y_{t+w_icandidate -1 } from the target time series
    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :param other_observed_vars:
    :param n_iidsamples: int number of observations (the length of the indices of time points so that it includes
    the oldest cause of target)
    :param p_MRplain:
    :return: p_MR
    """
    if not np.isnan(x1[other_observed_vars, :]).any():
        # if there are some variables in the conditioning set then use them for the conditional dependence test
        C = np.vstack((x1[i_candidate, :], Y_wi, x1[other_observed_vars, :], Y_wi_minus_1,
                       np.ones((1, n_iidsamples))))
        C = pd.DataFrame(C.T, columns = range(len(C)))
        out_stats = pingouin.partial_corr(C,0,1, list(range(2,len(C.T))), alternative='two-sided')
        rho_part_corr, p_part_corr = out_stats["r"], out_stats["p-val"]
        rho_MR, p_MR = rho_part_corr.values[0], p_part_corr.values[0]
        print(rho_MR)
        print(p_MR)
        print(out_stats)
        print(pingouin.correlation._correl_pvalue(rho_MR, len(C), len(C.columns)))
        #rho_MR_all, p_MR_all = partialcorrel(C.T)
        #p_MR = p_MR_all[0, 1]
        #rho_MR = rho_MR_all[0, 1]
    else:  # if the conditioning set does not include any variable then just use a plain dependency test
        p_MR = p_MRplain[i_candidate]
        rho_MR = 0
    return p_MR, rho_MR


def calculate_PM(xo: np.ndarray, x1: np.ndarray, relationships: str, i_candidate: int) -> np.ndarray:
    """
    Calculate the p value of the dependence X^i_{t-1} _|/|_ X_{t}
    Note: Based on our assumptions this is a redundant test. I just added it for sanity check


    :param xo: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t-1} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -2 step
    :param x1: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -1 step (these are the ones that belong to the S^i
     set of the candidate at this observation
    :param relationships: string 'linear' (implemented so far)
    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :return: p_PM
    """
    if relationships == 'linear':
        if np.array_equal(xo[i_candidate, :], xo[i_candidate, :].astype(bool)):
            rho_PM, p_PM = scipy.stats.pointbiserialr(xo[i_candidate, :], x1[i_candidate, :])
        else:
            rho_PM, p_PM = scipy.stats.pearsonr(xo[i_candidate, :], x1[i_candidate, :])
    return p_PM


def calculate_PMR(x1: np.ndarray, xo: np.ndarray, i_candidate: int, other_observed_vars: np.ndarray, n_iidsamples: int, Y_wi: np.ndarray, Y_wi_minus_1: np.ndarray) -> np.ndarray:
    """
    Calculate the p value of the conditional independence X^i_{t-1} _|/|_ Y_{t+wi} | {X^i_t, S^i, Y_{t+wi-1}}
    Condition 2 on paper

    :param x1: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -1 step (these are the ones that belong to the S^i
     set of the candidate at this observation
    :param xo: 2D np.ndarray (n_vars) x (n_time_steps): contains X^i_candidate_{t-1} at each t and for j != i_candidate
    contains the nodes of the other time series t+w_icandidate -wj -2 step
    :param i_candidate: int index the time series under consideration (X^i in the paper) to decide if it is a cause
     of Y_{t+wi} or not
    :param other_observed_vars: 1D np.ndarray the indices of the time series j != {i_candidate, target}
    :param n_iidsamples: int number of observations (the length of the indices of time points so that it includes the oldest cause of target)
    :param Y_wi: 1D np.ndarray containing at each point the node Y_{t+w_icandidate} from the target time series
    :param Y_wi_minus_1: 1D np.ndarray containing at each point the node Y_{t+w_icandidate -1 } from the target time series
    :return: p_PRM
    """

    if not np.isnan(x1[other_observed_vars,
                    :]).any():  # if there are some variables in the conditioning set then use them for the conditional dependence test
        C = np.vstack((xo[i_candidate, :], Y_wi, x1[other_observed_vars, :],
                       x1[i_candidate, :], Y_wi_minus_1, np.ones((1, n_iidsamples))))
    else:  # if the conditioning set does not include any variable then just use only X_t for the conditioning variable
        C = np.vstack((xo[i_candidate, :], Y_wi, x1[i_candidate, :], Y_wi_minus_1,
                       np.ones((1, n_iidsamples))))
    
    C = pd.DataFrame(C.T, columns = range(len(C)))
    out_stats = pingouin.partial_corr(C,0,1, list(range(2,len(C.T))), alternative='two-sided')
    rho_part_corr, p_part_corr = out_stats["r"], out_stats["p-val"]
    p_PRM = p_part_corr
    #rho_part_corr, p_part_corr = partialcorrel(C.T)
    #p_PRM = p_part_corr[0, 1]
    return p_PRM


def list_diff(list1, list2):
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)
    return out


def SyPI_method(regression_algorithm: str, relationships: str, normalized_data: bool, normalization: str, lambda_reg: float, order:int,  p_cond1: float, p_cond2: float, threshold_lasso: float,
         var_direct_causes: set, var_indirect_causes: set, var_non_causes: set, X: np.ndarray) -> Tuple[list, list,
                                                                                                        list, list,
                                                                                                        list, list,
                                                                                                        int, list]:
    """
    Detect direct and indirect causes of the last row of X, from the candidate time series in X

    :param regression_algorithm: string which algorithm to use for lag calculation.
    Implemented so far: lasso regression 'scipy_lassoalgo'
    :param relationships: string 'linear' (implemented so far)
    :param normalized_data: boolean True if want to normalize the data for the lag calculation, False otherwise
    :param normalization: type of normalization 'variance', 'minmax'
    :param lambda_reg: float lambda regularizer for the lasso regression
    :param order: int number of lags to take into account when computing w_max with lasso.
    :param p_cond1: float threshold1 for conditional dependence
    :param p_cond2: float threshold2 for conditional independence
    :param threshold_lasso: float threshold for lasso coefficients
    :param var_direct_causes: set of ground truth direct causes for calculation of false positives, false negatives,
     true positives, true negatives
    :param var_indirect_causes: set of ground truth indirect causes for calculation of false positives, false negatives,
     true positives, true negatives
    :param var_non_causes: set of ground truth non causes for calculation of false positives, false negatives,
     true positives, true negatives
    :param X: 2D numpy array (number of time series) x (number of time steps).
    Note the last row must always correspond to the target time series
    :return: predicted_causes, predicted_non_causes, false_positives_predict_direct_and_indirect,
    true_positives_predict_direct_and_indirect, false_negatives_predict_direct_and_indirect,
    true_negatives_predict_direct_and_indirect, n_vars
    """
    n_allvars = np.shape(X)[0]
    n_vars = n_allvars - 1
    n_time_steps = np.shape(X)[1]

    # calculate the lag between each time series and the target
    w_max, w = calculate_w_wmax(X, threshold_lasso, lambda_reg, regression_algorithm, normalized_data, normalization, order)

    predicted_causes = []
    strength_of_predicted_causes = []
    for i_candidate in range(n_vars):
        if not np.isnan(w[i_candidate]):
            xo, x1, Y_wi_minus_1, Y_wi = calculate_conditioning_set_and_vars(i_candidate, X, n_time_steps, w, w_max, n_vars)
            n_iidsamples = np.shape(Y_wi_minus_1)[0]
            p_MRplain = calculate_pMRplain(n_vars, x1, Y_wi, xo, w)
            if relationships == 'linear':
                if p_MRplain[i_candidate] < p_cond1:
                    try:
                        other_observed_vars = np.setdiff1d(np.where(p_MRplain < p_cond1), i_candidate)
                        # only those associated with R to get rid of redundant conditioning
                    except:
                        other_observed_vars = []
                    p_MR, rho_MR = calculate_pMR(x1, Y_wi, Y_wi_minus_1, i_candidate, other_observed_vars, n_iidsamples, p_MRplain)
                    if p_MR < p_cond1:  # and rho_MR > 0.1:
                        p_PM = calculate_PM(xo, x1, relationships, i_candidate)
                        if p_PM < p_cond1:  # and rho_PM > 0.1:
                            if relationships == 'linear':
                                p_PRM = calculate_PMR(x1, xo, i_candidate, other_observed_vars, n_iidsamples, Y_wi, Y_wi_minus_1)
                            if p_PRM > p_cond2:
                                predicted_causes.append(i_candidate)
                                strength_of_predicted_causes.append(rho_MR)

    predicted_non_causes = list_diff(list(range(n_vars)), predicted_causes)
    
    if var_direct_causes is None:
        return predicted_causes
        
        
    if var_direct_causes == [] and var_indirect_causes == [] and var_non_causes == []:
        print('...No ground truth is provided. I cannot calculate False positives and False negatives...')
        false_positives_predict_direct_and_indirect = []
        false_negatives_predict_direct_and_indirect = []
        true_positives_predict_direct_and_indirect = []
        true_negatives_predict_direct_and_indirect = []
    else:
        #false_positives_predict_direct_and_indirect, true_positives_predict_direct_and_indirect = calculate_false_positives(
        #    (var_direct_causes | var_indirect_causes), predicted_causes)
        #false_negatives_predict_direct_and_indirect, true_negatives_predict_direct_and_indirect = calculate_false_negatives(
        #    var_non_causes, predicted_non_causes)
        pass

    return predicted_causes, predicted_non_causes, false_positives_predict_direct_and_indirect,\
           true_positives_predict_direct_and_indirect, false_negatives_predict_direct_and_indirect,\
           true_negatives_predict_direct_and_indirect, n_vars, strength_of_predicted_causes
