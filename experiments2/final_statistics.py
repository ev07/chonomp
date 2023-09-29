





def extract_best_configuration(cls_name, fs_name, filename, metric):
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename, header=True)
    df_params = pd.read_csv(params_dir+filename, header=True)
    
    df_params = df_params[df_params[cls_col]==cls_name & df_params[fs_col]==fs_name][[key1,key2]]
    df_bootstrap = df_bootstrap[[key1, key2, metric]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])
    
    #at this point, the selected entries correspond to only the right cls and fs.
    
    df_
