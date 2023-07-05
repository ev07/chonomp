import sys
import os

from darts.models import TFTModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from torch.nn import MSELoss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd

rootdir = '../'
sys.path.append(rootdir)

import data_opener



def build_TFT(config):
    model = TFTModel(**config)
    return model

def fit_TFT(model, train_data, target, causal=False):
    scalerY = StandardScaler()
    scalerCov = StandardScaler()
    
    train_series = train_data[[target]]
    covariates = train_data[[x for x in train_data.columns if x!=target]]
    
    train_series = scalerY.fit_transform(train_series)
    covariates = scalerCov.fit_transform(covariates)
    
    if causal:
        #shift covariates by one in the past, so the input at time t is cov_{t-1}
        train_series = train_series[1:]
        covariates = covariates[:len(covariates)-1]
    
    train_series = TimeSeries.from_values(train_series)
    covariates = TimeSeries.from_values(covariates)
    
    model.fit(train_series, future_covariates=covariates, verbose=0)
    
    return scalerY, scalerCov
    
    
    
def predict_TFT(model, val_data, scalerY, scalerCov, target, input_chunk_length, forecast_horizon, causal=False):
    val_series = val_data[[target]]
    covariates = val_data[[x for x in val_data.columns if x!=target]]
    
    val_series = scalerY.fit_transform(val_series)
    covariates = scalerCov.fit_transform(covariates)
    
    new_index = list(range(input_chunk_length,len(val_data)-forecast_horizon-1))
    
    if causal:
        #shift covariates by one in the past, so the input at time t is cov_{t-1}
        val_series = val_series[1:]
        covariates = covariates[:len(covariates)-1]
        new_index = new_index[1:]
    
    
    bigger_series = [val_series[i-input_chunk_length:i] for i in new_index]
    bigger_future = [covariates[i-input_chunk_length:i+forecast_horizon] for i in new_index]
    
    
    bigger_series = [TimeSeries.from_values(x) for x in bigger_series]
    bigger_future = [TimeSeries.from_values(x) for x in bigger_future]
    
    res = model.predict(1, bigger_series,
                     future_covariates=bigger_future,
                     roll_size = 1,
                     verbose=0
                     )
                     
    res = np.array([x.values() for x in res])
    res = res.reshape((-1,1))
    
    res = scalerY.inverse_transform(res)
    
    res = pd.DataFrame(res,index=val_data.index[new_index], columns =[target])
    
    return res


def complete_config(config):
    config_ref = {
        "input_chunk_length":10,
        "output_chunk_length":1,
        "hidden_size":64,
        "lstm_layers":1,
        "num_attention_heads":4,
        "dropout":0.1,
        "likelihood":None,
        "loss_fn":MSELoss(),
        "batch_size":32,
        "n_epochs":3,
        "add_relative_index":False,
        "add_encoders":None,
        "random_state":42,
        }
    for key in config_ref:
        if key not in config:
            config[key] = config_ref[key]
    return config

def compute_metrics(y_pred, y_true):
    y_true = y_true[y_pred.index]  # align time series
    
    statistics = dict()
    statistics["sse"] = np.sum((y_true - y_pred)**2)
    statistics["R2"] = 1-statistics["sse"]/np.std(y_true)**2/len(y_true)
    statistics["rmse"] = np.sqrt(statistics["sse"]/len(y_true))
    statistics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    statistics["mfe"] = np.mean(y_true - y_pred)
    return statistics
    

def TFT_baseline(data_train, data_val, target,  config, causal=True):
    """
    Main function to be called.
    Providing the train and val/test dataset, a target and the model config,
    it returns several test metrics at the end of the training.
    If causal is true, the covariates will be shifted by one as the model originally uses lag0 covariates.
    """

    config = complete_config(config)
    model = build_TFT(config)
    
    scalerY, scalerCov = fit_TFT(model, data_train, target, causal=causal)
    
    fittedvalues = predict_TFT(model, data_val, scalerY, scalerCov, target, config["input_chunk_length"], config["output_chunk_length"], causal=causal)
    
    metrics = compute_metrics(fittedvalues[target], data_val[target])
    
    return metrics


if __name__=="__main__":
    
    dataset = "SynthNonlin/7ts2h"
    filename = "data_0.csv"
    data, var_names, causes = data_opener.open_dataset_and_ground_truth(dataset, filename, rootdir=rootdir)
    
    data = data.reset_index(drop=True)  # make sure we get a RangeIndex compatible index
    train, val = data.loc[:int(len(data)*0.7)], data.loc[int(len(data)*0.7):]
    

    config = complete_config(dict())
    
    stats = TFT_baseline(train, val, "E", config)
    
    print(stats)
    
    
    
    
    
    
