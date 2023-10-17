import pandas as pd





def extract_mean_best_configuration(df_select: pd.DataFrame, metric: str, direction="max"):
    """
    For a given pair (classifier, fs), gives back the best in score and corresponding out_score
    direction states whether the metric should be maximized or minimized
    
    params:
        df_select: dataframe, contains the rows associated to the chosen cls, fs algorithms.
            It has to contain at least three columns: 'fold', 'in_'+metric, 'out_'+metric
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    folds = "fold"
    
    if direction=="max":
        df_best_rows = df_select.loc[df_select.groupby(by=[folds])["in_"+metric].idxmax()]
    else:
        df_best_rows = df_select.loc[df_select.groupby(by=[folds])["in_"+metric].idxmin()]
        
    final_perf = df_best_rows["out_"+metric].mean()
    
    return final_perf

def extract_mean_best_configuration_from_file(cls_name: str, fs_name:str , filename: str, metric: str, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    extract rows corresponding to a given CLS and FS algorithm, 
    and compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        cls_name: name of the classifier whose hyperparameters are optimized
        fs_name: name of the feature selection algorithm whose hyperparameters are optimized
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    df_params = df_params[df_params[cls_col]==cls_name & df_params[fs_col]==fs_name][[key1,key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])

    final_perf = extract_best_configuration(df_select, metric, direction=direction)
    
    return final_perf
    
    
    
def all_methods_mean_best_configuration(filename, metric, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        cls_name: name of the classifier whose hyperparameters are optimized
        fs_name: name of the feature selection algorithm whose hyperparameters are optimized
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])
    
    #for each fold in each model type, select the best configuration score
    if direction=="max":
        df_best_rows = df_select.loc[df_select.groupby(by=[folds, cls_col, fs_col])["in_"+metric].idxmax()]
    else:
        df_best_rows = df_select.loc[df_select.groupby(by=[folds, cls_col, fs_col])["in_"+metric].idxmin()]
    
    print(df_best_rows)
    #average score across each model type
    df_metric = df_best_rows.groupby(by=[cls_col, fs_col])["out_"+metric].mean().reset_index()
    
    return df_metric
    


    
    
    
def all_methods_best_mean_configuration(filename, metric, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        cls_name: name of the classifier whose hyperparameters are optimized
        fs_name: name of the feature selection algorithm whose hyperparameters are optimized
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])
    
    #mean of each config across all folds
    df_mean_rows = df_select.groupby(by=[key1, key2, cls_col, fs_col])["in_"+metric].mean().reset_index()
    
    #select best configs for each model type
    if direction=="max":
        df_best_config = df_mean_rows.loc[df_mean_rows.groupby(by=[cls_col, fs_col])["in_"+metric].idxmax()]
    else:
        df_best_config = df_mean_rows.loc[df_mean_rows.groupby(by=[cls_col, fs_col])["in_"+metric].idxmin()]
    
    #merge to select only best configs
    df_all_folds = df_best_config[[cls_col, fs_col, key1, key2]].merge(df_select, on=[key1, key2, cls_col, fs_col])
    
    #compute mean across folds of best configs.
    df_metric = df_all_folds.groupby(by=[cls_col, fs_col, key1, key2])["out_"+metric].mean().reset_index()
    
    return df_metric, df_all_folds[[key1, "in_"+metric,"out_"+metric, folds]]
    
    
    
    
    
    
    
    
    
if __name__=="__main__":
    print(all_methods_mean_best_configuration("pairwiseLinear_data_0_X666.csv", "rmse", direction="min"))
    print(all_methods_best_mean_configuration("pairwiseLinear_data_0_X666.csv", "rmse", direction="min"))
    


