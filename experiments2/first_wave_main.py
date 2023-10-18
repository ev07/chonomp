import sys
import os
import time
import pandas as pd
import numpy as np

import optuna
from optuna.samplers import RandomSampler

from routines import full_experiment as launch_experiment

rootdir = '../'
sys.path.append(rootdir)

import baselines.estimators
import baselines.feature_selection
from data_opener import open_dataset_and_ground_truth
from models import NotEnoughDataError

optuna.logging.set_verbosity(optuna.logging.WARNING)


def save_append(df, path):
    df_0 = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    df = pd.concat([df_0, df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


def setup_dataset(dataset_name, filename, target):
    if dataset_name == "VAR10":
        config = {"DATASET":{
            "PATH": "VARSmall/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "all",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "pairwiseLinear":
        config = {"DATASET":{
            "PATH": "dgp/piecewise_linear/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "monotonic":
        config = {"DATASET":{
            "PATH": "dgp/monotonic/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "trigonometric":
        config = {"DATASET":{
            "PATH": "dgp/trigonometric/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "wikipedia":
        config = {"DATASET":{
            "PATH": "wikipediaMathEssencials/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 100}
            }
    elif dataset_name == "Appliances":
        config = {"DATASET":{
            "PATH": "Appliances",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "AusMacro":
        config = {"DATASET":{
            "PATH": "AusMacro",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "AusMeteo":
        config = {"DATASET":{
            "PATH": "AusMeteo",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 100}
            }
    config["DATASET"]["NAME"] = dataset_name
    config["DATASET"]["FILENAME"] = filename
    config["DATASET"]["TARGET"] = target
    config["DATASET"]["HOLDOUT"] = True
    config["DATASET"]["HOLDOUT_SIZE"] = 50
    return config


def setup_config(trial, fs_name, cls_name):
    cls_conf = baselines.estimators.generate_optuna_parameters(cls_name, trial)
    cls_conf = baselines.estimators.complete_config_from_parameters(cls_name, cls_conf)
    fs_conf = baselines.feature_selection.generate_optuna_parameters(fs_name, trial)
    fs_conf = baselines.feature_selection.complete_config_from_parameters(fs_name, fs_conf)
    

    config = {"CLS":{"NAME":cls_name,
           "CONFIG":cls_conf},
     "FS":{"NAME":fs_name,
           "CONFIG":fs_conf},
     "FOLDS":{"NUMBER_FOLDS": 5,
              "WINDOW_SIZE": 0.4,
              "STRATEGY": "fixed_start"}
    }
    return config




def generate_optuna_objective_function(fs_name, cls_name, dataset_setup, objective="R2"):
    memorize = {"params":[], "results":[]}
    def optuna_objective_function(trial):
        config_file = setup_config(trial, fs_name, cls_name)
        config_file = {**config_file, **dataset_setup}
        
        config_name = fs_name + "_" + cls_name + "_" + str(trial.number)
        
        # sometimes, the configuration is not adapted to the model, and will return an error.
        # rather than check the validity of every configuration, it is simple to catch the exceptions
        # coming from such configurations.
        try:
            df_results, df_params = launch_experiment(config_file, config_name, run_bootstrap=False, compute_selected_stats=True)
        except NotEnoughDataError as e:
            print("Configuration",trial.number,"failed with error:\n",str(e))
            raise e
            return np.inf
        
        # save resulting data
        memorize["params"].append(df_params)
        memorize["results"].append(df_results)
        
        return df_results[objective].mean()
        
    return optuna_objective_function, memorize
    
    
    
    return df_bootstrap[objective].mean()

def full_experiment(dataset, fs_name, cls_name, seed=0):
    ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
    data_dir = ex_datasetup["PATH"]
    target_extraction = ex_datasetup["TARGET_CHOICE"]
    maximum_target_extraction = ex_datasetup["MAXIMUM_NUMBER_TARGETS"]
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    filelist = list(os.listdir(rootdir + "data/" + data_dir + "/"))
    
    first_evaluation_flag = True  # flag for the first call to objective, to get time estimation.
    
    for i,filename in enumerate(filelist):
        if not os.path.isfile(rootdir + "data/" + data_dir + "/" + filename):
            continue
        print("New file, time since begining is", time.time()-start_time, "(", i, "/", len(filelist), ")")
        
        data, var, _, _ = open_dataset_and_ground_truth(data_dir, filename, "parents", rootdir)
        # make sure to avoid extracting all targets in large datasets
        if target_extraction == "all":
            target_set = var
        elif target_extraction == "sampling":
            target_set = rng.choice(var,size=(maximum_target_extraction,), replace=False, shuffle=False)
        elif target_extraction == "given":
            target_set = var
            
        for target in target_set:
            print("\tTarget", target, "beggining")
            
            if first_evaluation_flag: # record time on first iteration
                time_begin = time.time()
            
            dataset_setup = setup_dataset(dataset, filename, target)
        
            objective, results = generate_optuna_objective_function(fs_name, cls_name, dataset_setup)
            study = optuna.create_study(sampler=RandomSampler())
            study.optimize(objective, n_trials=100)
            
            #results contains all the training information from the trials
            filename_xt = os.path.splitext(filename)[0]
            
            params_df = pd.concat(results["params"])
            results_df = pd.concat(results["results"])
            bootstrap_df = pd.concat(results["bootstrap"])
            
            save_append(params_df,"./results/optuna/params/"+dataset+"_"+filename_xt+"_"+target+".csv")
            save_append(results_df,"./results/optuna/test_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            #save_append(bootstrap_df,"./results/optuna_bootstrap/bootstrap_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            
            if first_evaluation_flag:  # give estimation of the procedure length in seconds
                first_evaluation_flag=False
                t = time.time()-time_begin
                print("\t\tOne variable took",t ,"seconds.")
                print("\t\tEstimated time for whole pipeline:",len(target_set)*len(filelist)*t, "seconds")



if __name__=="__main__":
    full_experiment("VAR10", "ChronOMP", "SVRModel")



    
