import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import pickle
import gzip
import time

import captum
import shap


sys.path.append("../")
from final_statistics import *
from results_other import get_splitted_dataset, setup_dataset, get_best_hyperparameters_from_keys, hyperparam_setting
from baselines.estimators import TFTModel, SVRModel, ARDLModel, KNeighborsRegressorModel
from data_opener import open_dataset_and_ground_truth
from baselines.feature_selection import ChronOMP, GroupLasso, NoSelection

import shap








target = "0"
dataset = "NoisyVAR_8000"
_, filename = sys.argv
seed = int(filename.split(".")[0].split("_")[1])

result_filename = dataset+"_"+os.path.splitext(filename)[0]+"_"+target+".csv"

results_dir = "../experiments2/results/optuna/test_stats/"
params_dir = "../experiments2/results/optuna/params/"
save_dir = "./results/shap_explain_uniqueMB/"+dataset+"/"

fs_version  ="NoSelection"
cls_version = "SVRModel"

key1 = "config_name"
key2 = "start_time"
cls_col = "CLS.NAME"
fs_col = "FS.NAME"

df_results = pd.read_csv(results_dir+result_filename)
df_params = pd.read_csv(params_dir+result_filename)
df_params = df_params[[cls_col, fs_col, key1, key2]]
df_results = df_results[[key1, key2, "R2", "FS_time"]]
df_select = df_params.merge(df_results, on=[key1, key2])
df_best_config = df_select.loc[df_select.groupby(by=[cls_col, fs_col])["R2"].idxmax()]

ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
data_dir = ex_datasetup["PATH"]







data, _,_, _ = open_dataset_and_ground_truth(data_dir, filename, "parents", "../")
data_train, data_test = get_splitted_dataset(data, dataset, 1.0)










best_hyperparams = get_best_hyperparameters_from_keys(result_filename, df_best_config)

best_hyperparams_algo = hyperparam_setting(best_hyperparams, fs_version, cls_version)

config = best_hyperparams_algo["FS"]["CONFIG"] if "CONFIG" in best_hyperparams_algo["FS"] else dict()
fs = {"NoSelection":NoSelection}[fs_version](config, target)  # only NoSelection should be used for this experiment
fs.fit(data_train)
fs_set = fs.get_selected_features()
if target not in fs_set:
    fs_set.append(target)


# Selection step for unique MB
df = pd.read_csv("../data/equivalence_datasets/NoisyVAR_characteristics.csv")
th_solution = df[df["filename"]==filename]
th_solution = th_solution["theoretical_solution"]
th_solution = eval(th_solution.values[0])
th_solution = [list(map(str, l)) for l in th_solution]

for equiv_set in th_solution:
    if len(equiv_set)==1:
        continue
    for variable in equiv_set[1:]:
        fs_set.remove(variable)






data_train_selected = data_train[fs_set]
data_test_selected = data_test[fs_set]





forecaster_hp = best_hyperparams_algo["CLS"]["CONFIG"]
forecaster_constructor = {"TFTModel":TFTModel,"SVRModel":SVRModel,
                           "ARDLModel":ARDLModel,
                           "KNeighborsRegressorModel":KNeighborsRegressorModel}[cls_version]
forecaster = forecaster_constructor(forecaster_hp, target)
predictedvalues = forecaster.fit_predict(data_train_selected, data_test_selected)

metrics = forecaster.compute_metrics(predictedvalues, data_test_selected[target])
f = open(save_dir+"metrics/"+fs_version+"_"+cls_version+"_"+filename+".pkl","wb")
pickle.dump(metrics, f)
f.close()

columns = list(data_test_selected.columns)
f = open(save_dir+"columns/"+fs_version+"_"+cls_version+"_"+filename+".pkl","wb")
pickle.dump(columns, f)
f.close()





X, y, indices = forecaster.prepare_data_vectorize(data_test_selected)

shap_values_all = []
batch_size = 1

count = 0
rng = np.random.default_rng(seed)
for i in list(range(0,len(X),batch_size)):
    count+=1
    if count>20: #limit size due to time constraints. 10 samples take 70 secs approx.
        break
        
    Xbatch = X[i:i+batch_size]

    explainer = shap.KernelExplainer(forecaster.model.predict, np.zeros(Xbatch.shape))

    shap_values = explainer.shap_values(Xbatch, silent=True, nsamples = Xbatch.shape[1])
    
    shap_values_all.append(shap_values[0])

shap_values_all = np.array(shap_values_all)

f = gzip.open(save_dir+"explanations/"+fs_version+"_"+cls_version+"_"+filename+".pkl","wb")
pickle.dump(shap_values_all, f)
f.close()



















