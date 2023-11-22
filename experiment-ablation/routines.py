import sys
import os

import argparse    
import yaml
import pandas as pd
import time
import numpy as np

rootdir = '../'
sys.path.append(rootdir)

from baselines.estimators import Estimator, ARDLModel, SVRModel, KNeighborsRegressorModel
from baselines.feature_selection import ChronOMP, BivariateGranger, ModifiedRFE, VectorLassoLars, BackwardChronOMP

from data_opener import open_dataset_and_ground_truth

class TargetNotSelectedError(ValueError):
    pass




def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
def flatten_dict(d, name=""):
    flat = pd.json_normalize(d).to_dict(orient="records")[0]
    if name!="":
        flat = {name+"."+key: value for key, value in flat.items()}
    return flat
        

def get_FS(config_file):
    fs_info = config_file["FS"]
    config = fs_info["CONFIG"]
    
    algo = {"ChronOMP":ChronOMP,
            "BackwardChronOMP":BackwardChronOMP,
            "BivariateGranger":BivariateGranger,
            "VectorLassoLars":VectorLassoLars,
            "ModifiedRFE":ModifiedRFE}[fs_info["NAME"]]
    
    constructor = lambda target: algo(config, target)
    descriptors = flatten_dict(fs_info,name="FS")
    descriptors["SELECTION_TYPE"] = algo.selection_mode
    return constructor, descriptors

def get_CLS(config_file):
    cls_info = config_file["CLS"]
    config = cls_info["CONFIG"]
    name = cls_info["NAME"]
    model = {
    "ARDLModel":ARDLModel, 
    "SVRModel": SVRModel, 
    "KNeighborsRegressorModel": KNeighborsRegressorModel
    }[name]
    
    constructor = lambda target: model(config, target)
    #flatten the dictionnary
    descriptors = flatten_dict(cls_info, name="CLS")
    
    return constructor, descriptors


    

def data_generator_main(config_file, rootdir="../", seed=0, ):
    """
    Generator function that either go through the whole data folder,
    or just one file and just one target.
    The configuration also has the ability to restrict the sample size and number of covariates.
    Params:
        config_file: dictionary containing the whole experimental config. Must contain:
            "DATASET": dictionary:
                "PATH": string, path of the dataset within the ./data/ folder
                "NAME": string, dataset name
                "CAUSES": string, what cause extraction strategy to use
                "TARGET_CHOICE": string, how to choose the targets among the available variables
                "MAXIMUM_NUMBER_TARGETS": int, bound on the maximal number of target per file.
                (optional) "FILENAME": if a specific file alone must be run, name of this file.
                (optional) "TARGET": if a specific target alone must be run, name of this target.
                (optional) "HOLDOUT": bool, set to True to keep a hold out set at the end of the dataset.
                (optional) "HOLDOUT_RATIO": float, fraction of observations to keep aside for the holdout set. Applied before SAMPLE_RATIO. Size of Tuning set is datasize * SAMPLE_RATIO * (1-HOLDOUT_RATIO) unless the holdout size is less than 50, otherwise a holdout set of 50 is kept.
                (optional) "SAMPLE_RATIO": float, fraction of the original tuning dataset timesteps to keep as tuning dataset. Useful for data size ablation studies
                (optional) "COVARIATE_RATIO": float, fraction of the covariates to do the experiment on. Useful for data size ablation studies
        rootdir (optional): str, path to root of the project (where to find ./data/)
        seed (optional): int, seed for the random sampling of variables.
        
    Returns:
        data: the dataframe containing the data
        target: name of the variable to predict
        nonlagged causes: list of relevant variables in GT, or None
        lagged causes: list of relevant (variables, lags) in GT, or None
        descriptors: data descriptors for logging
    """
    data_info = config_file["DATASET"]
    data_dir = data_info["PATH"]
    data_name = data_info["NAME"]
    cause_extraction = data_info["CAUSES"]
    target_extraction = data_info["TARGET_CHOICE"]
    maximum_target_extraction = data_info["MAXIMUM_NUMBER_TARGETS"]
    holdout = False if "HOLDOUT" not in data_info else data_info["HOLDOUT"]
    holdout_ratio = 0. if "HOLDOUT_RATIO" not in data_info else data_info["HOLDOUT_RATIO"]
    sample_ratio = 1. if "SAMPLE_RATIO" not in data_info else data_info["SAMPLE_RATIO"]
    covariate_ratio = 1. if "COVARIATE_RATIO" not in data_info else data_info["COVARIATE_RATIO"]
    
    filelist = [data_info["FILENAME"]] if "FILENAME" in data_info else os.listdir(rootdir + "data/" + data_dir + "/")
    
    rng = np.random.default_rng(seed)
    
    
    for filename in filelist:
        if not os.path.isfile(rootdir + "data/" + data_dir + "/" + filename):
            continue
        
        data, var, causes, lagged_causes = open_dataset_and_ground_truth(data_dir, filename, cause_extraction, rootdir)
        
        
        # remove hold out test set if given
        if holdout:
            holdout_size = int(holdout_ratio*len(data))
            if holdout_size < 50:
                holdout_size = 50
            data = data.iloc[:-holdout_size]
        
            # remove data steps when doing ablation studies
            new_size = int(len(data)*sample_ratio)
            data = data.iloc[:new_size]
       
        
        # make sure to avoid extracting all targets in large datasets
        if target_extraction == "all":
            target_set = var
        elif target_extraction == "sampling":
            target_set = rng.choice(var,size=(maximum_target_extraction,), replace=False, shuffle=False)
        elif target_extraction == "given":
            target_set = var
            
        #handle single target given in config file
        target_list = [data_info["TARGET"]] if "TARGET" in data_info else target_set
        
        for target in target_list:
        
            covariate_size = len(data.columns)
            if covariate_ratio != 1.:  # change the number of covariates for ablation studies.
                to_exclude = causes[target] if target in causes[target] else causes[target]+[target]  # keep target also
                to_include = [col for col in data.columns if col not in to_exclude]
                covariate_size = int(covariate_ratio*len(to_include))
                chosen_columns = list(rng.choice(to_include, size = (covariate_size,), replace=False, shuffle=False)) + list(to_exclude)
                covariate_size = len(chosen_columns)
                data = data[chosen_columns]
        
            descriptors = {"data_name": data_name, "filename": filename,"target": target}
            if not holdout:
                descriptors = {**descriptors,
                           "holdout": max(int(holdout_ratio*len(data)), 50),
                           "effective_training_size": int((len(data) - max(int(holdout_ratio*len(data)), 50))*sample_ratio),
                           "covariate_size": covariate_size}
            
            
            if causes is not None:  # ground truth not in dataset
                yield data, target, causes[target], lagged_causes[target], descriptors
            else:
                yield data, target, None, None, descriptors

    
    

def get_folds_from_data(data, config_file):
    # apply the FCCV approach
    config = config_file["FOLDS"]
    data_config = config_file["DATASET"]
    numberfolds = config["NUMBER_FOLDS"]
    windowsize = config["WINDOW_SIZE"]
    sample_ratio = 1. if "SAMPLE_RATIO" not in data_config else data_config["SAMPLE_RATIO"]
    
    if windowsize>0 and windowsize<=1:  # Float given
        windowsize = int(windowsize*len(data))  # the starting window size for the train part
    if numberfolds>0:
        windowsize = int(windowsize*sample_ratio)  # decrease the window size proportionally.
        datalen = int(len(data)*sample_ratio)  # change data overall size so that sample size miniaturizes all training sets.
        data = data.iloc[:datalen]
    elif numberfolds==0: # Case for the holdout set
        holdout_size = len(data) - windowsize
        if holdout_size < 50:
            holdout_size = 50
            windowsize = len(data) - holdout_size
        windowsize = int(windowsize*sample_ratio)
        numberfolds = 1
        
    strategy = config["STRATEGY"]
    
    
    if strategy == "fixed_start":
        for fold in range(numberfolds):
            if config["NUMBER_FOLDS"] == 0:  # test part, fixed split
                yield data.iloc[0:windowsize,:], data.iloc[-holdout_size:,:]
            else:
                start = 0
                middle = windowsize + int(((len(data)-windowsize)/(numberfolds))*fold)
                end = windowsize + int(((len(data)-windowsize)/(numberfolds))*(fold+1))
                if end>len(data):
                    end=len(data)  # in case of integer division
                yield data.iloc[start:middle,:], data.iloc[middle:end,:]
    elif strategy == "rolling":
        for fold in range(numberfolds):
            if config["NUMBER_FOLDS"] == 0:  # test part, fixed split
                yield data.iloc[0:windowsize,:], data.iloc[-holdout_size:,:]
            else:
                start = int(((len(data)-windowsize)/(numberfolds))*fold)
                middle = windowsize + int(((len(data)-windowsize)/(numberfolds))*fold)
                end = windowsize + int(((len(data)-windowsize)/(numberfolds))*(fold+1))
                if end>len(data):
                    end=len(data)  # in case of integer division
                yield data.iloc[start:middle,:], data.iloc[middle:end,:]
    
    

def compute_stats_selected(totalcolumns, selected, causes, lagged_causes, selection_mode):
    row = {"FS_size":len(selected)}
    if causes is not None:  # if none, the data has no ground truth graph. So no stats to compute.
        if selection_mode == "variable":
            sPred = set(selected)
            sTrue = set(causes)
        elif selection_mode == "variable, lag":
            # assume output is of form (variable, lag)
            sPred = set(map(str,selected))
            sTrue = set(map(str,lagged_causes))
        else:
            raise NotImplementedError("Feature Selection algorithm has unrecognized selection mode (see FeatureSelector base class)")
        sTP = sTrue & sPred
        recall = len(sTP)/len(sTrue) if len(sTrue)>0 else 0
        precision = len(sTP)/len(sPred) if len(sPred)>0 else 0
        row = {**row, 
               "precision": precision,
               "recall": recall,
               "TP": len(sTP), 
               "FP": len(sPred) - len(sTP),
               "FN": len(sTrue) - len(sTP), 
               "TN": totalcolumns + len(sTP) - len(sPred) - len(sTrue)}
    return row
    

def compute_metrics(y_pred, y_true):
    return Estimator(None, None).compute_metrics(y_pred, y_true)
def compute_bootstrap_metrics(y_pred, y_true):
    return Estimator(None, None).compute_BBCCV(y_pred, y_true)


def full_experiment(config_file, config_name, run_bootstrap=False, compute_selected_stats=False, return_selected=False):
    start_time = time.time()
    rootdir = "../"

    # prepare objects and configuration descriptors

    results = []
    experiment_descriptors = []
    bootstrap_results = []
    
    FS_instance_generator, FS_descriptors = get_FS(config_file)
    CLS_instance_generator, CLS_descriptors = get_CLS(config_file)
    # folds and data preparation is an experiment-wide setting, so descriptors should be general
    expe_data_descriptors = flatten_dict(config_file["DATASET"], name="DATASET")
    folds_descriptors = flatten_dict(config_file["FOLDS"], name="FOLDS")
    
    for data, target, causes, lagged_causes, data_descriptors in data_generator_main(config_file, rootdir=rootdir):
    
        # SEPARATE into folds
        #
        folds = get_folds_from_data(data, config_file)
        
        test_fittedvalues = []
        test_truevalues = []
        selected_statistics = []
        selected = []
        
        for train, test in folds:
        
            # ESTIMATE FS algorithm on train
            # 
            t = time.time()
            fs_instance = FS_instance_generator(target)
            fs_instance.fit(train)
            selected = fs_instance.get_selected_features()
            t = time.time() - t
            
            
            if target not in selected: # target must be an input <- we consider autoregressive models
                selected.append(target)
            
            # COMPUTE stats for the selected set
            #
            if compute_selected_stats:
                selection_mode = fs_instance.selection_mode
                stats = compute_stats_selected(len(data.columns), selected, causes, lagged_causes, selection_mode)
                stats["FS_time"] = t
                selected_statistics.append(stats)
            
            # ESTIMATE classifier in train also, get fitted on test
            #
            t = time.time()
            cls_instance = CLS_instance_generator(target)
            if selection_mode == "variable":  # entire data columns selected
                fittedvalues  = cls_instance.fit_predict(train[selected], test[selected])
            elif selection_mode == "variable, lag" and cls_instance.support_feature_filtering:
                fittedvalues  = cls_instance.fit_predict(train, test, selected)
            else:
                raise NotImplementedError("Cannot use lag FS method with this classifier")
            t = time.time() - t
            stats["CLS_time"] = t
            
            test_fittedvalues.append(fittedvalues)
            test_truevalues.append(test[target])
        
        # COMPUTE mean aggregated selected statistics
        #
        if compute_selected_stats:
            selected_statistics = dict([(key,np.mean([s[key] for s in selected_statistics])) for key in selected_statistics[0]])
        if return_selected:
            selected_statistics["SelectedSet"] = selected
        
        # COMPUTE stats for the model
        #
        fittedvalues = pd.concat(test_fittedvalues)
        truevalues = pd.concat(test_truevalues)
        test_statistics = compute_metrics(fittedvalues, truevalues)
        
        # COMPUTE bootstrap stats if necessary
        if run_bootstrap:
            rows = compute_bootstrap_metrics(fittedvalues, truevalues)
            rows = [{**row, "config_name": config_name, "start_time": start_time} for row in rows]
            bootstrap_results = bootstrap_results + rows
        
        # PRODUCE result row
        #
        base_row = {**data_descriptors, "config_name": config_name, "start_time": start_time}
        stats_row = {**base_row, **selected_statistics, **test_statistics}
        results.append(stats_row)
        
    # RECORD hyperparameters in a dataframe
    base_row = {**CLS_descriptors, **FS_descriptors, "config_name": config_name, "start_time": start_time}
    base_row = {**base_row, **folds_descriptors}
    experiment_descriptors.append(base_row)
    
    # SAVE results
    df_results = pd.DataFrame.from_records(results)
    df_params = pd.DataFrame.from_records(experiment_descriptors)
    if run_bootstrap:
        df_bootstrap = pd.DataFrame.from_records(bootstrap_results)
        return df_params, df_bootstrap
        
    return df_results, df_params


if __name__=="__main__":
    # open config file

    parser = argparse.ArgumentParser()
    parser.add_argument("config_name")
    args = parser.parse_args()
    config_file = read_yaml("./configs/" + args.config_name + ".yaml")
    
    df_results, df_params = full_experiment(config_file, args.config_name)
    
    if not df_results.empty:
        df_results_0 = pd.read_csv("./results/" + "results" + ".csv") if os.path.isfile(
            "./results/" + "results" + ".csv") else pd.DataFrame()
        df_results = pd.concat([df_results_0, df_results], ignore_index=True, sort=False)
        df_results.to_csv("./results/" + "results" + ".csv", index=False)
        
    if not df_params.empty:
        df_params_0 = pd.read_csv("./results/" + "params" + ".csv") if os.path.isfile(
            "./results/" + "params" + ".csv") else pd.DataFrame()
        df_params = pd.concat([df_params_0, df_params], ignore_index=True, sort=False)
        df_params.to_csv("./results/" + "params" + ".csv", index=False)
    
