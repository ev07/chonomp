import pandas as pd
from collections import defaultdict
import networkx as nx


def open_dataset_and_ground_truth(dataset_name: str,
                                  filename: str,
                                  cause_extraction="parents",
                                  rootdir="."):
    """
    Open a file in a dataset family, where the ground truth is known:
    Params:
     - dataset_name: name of the dataset family
     - filename: name of the file containing the MTS instance
     - cause_extraction (optional): the method to compute the relevant variables
     - rootdir (optional): string indicating the root repository
    Returns
     - df: the dataframe containing the MTS
     - var_names: the list of attribute names
     - causes_attributes_dict: dictionnary associating each attribute to the list of its relevant predictors.
    """

    if dataset_name[:11] == "SynthNonlin":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df = df[df.columns[1:]]

    elif dataset_name[:4] == "fMRI":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]

    elif dataset_name[:10] == "FinanceCPT":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, header=None)
        df.columns = [str(i) for i in df.columns]

    elif dataset_name[:18] == "TestCLIM_N-5_T-250":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, header=None, sep=" ")
        df.columns = [str(i) for i in df.columns]
        
    elif dataset_name=="VARProcess/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARProcessNoCorr/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARLarge/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARSmall/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARVaried/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    else:
        raise Exception("Dataset specified in config file is not implemented")

    var_names = list(df.columns)

    if dataset_name[:11] == "SynthNonlin":
        if dataset_name == "SynthNonlin/7ts2h":
            ground_truth_parents = defaultdict(list)
            ground_truth_lags = 10  # could be anything since we don't care about lags in this project
            ground_truth_parents["A"] = [("D", 1), ("A", 1)] + [("B", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["D"] = [("H", 1), ("D", 1)] + [("E", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["H"] = [("C", 1), ("H", 1)]
            ground_truth_parents["C"] = [("C", 1)]
            ground_truth_parents["F"] = [("C", 1), ("F", 1)]
            ground_truth_parents["B"] = [("F", 1), ("B", 1)] + [("A", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["E"] = [("B", 1), ("E", 1)] + [("D", i) for i in range(1, ground_truth_lags + 1)]
        else:
            raise Exception("Dataset specified in argument is not implemented")

    elif dataset_name[:4] == "fMRI":
        index = filename[10:]
        index = index[:-4]
        g_truth_name = "fMRI_processed_by_Nauta/ground_truth/sim" + index + "_gt_processed.csv"

        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif dataset_name[:10] == "FinanceCPT":

        g_truth_name = "FinanceCPT/ground_truth/" + filename[:filename.find("_returns")] + ".csv"
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif dataset_name[:18] == "TestCLIM_N-5_T-250":
        g_truth_name = "TestCLIM_N-5_T-250/estimated_ground_truth/" + filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 2
        for cause, effect in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), 1))
            ground_truth_parents[str(effect)].append((str(cause), 2))
    
    elif dataset_name=="VARProcess/returns":
        g_truth_name = "VARProcess/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARProcessNoCorr/returns":
        g_truth_name = "VARProcessNoCorr/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARLarge/returns":
        g_truth_name = "VARLarge/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
    
    elif dataset_name=="VARSmall/returns":
        g_truth_name = "VARSmall/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARVaried/returns":
        g_truth_name = "VARVaried/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
    
    else:
        raise Exception("Dataset specified in argument is not implemented")

    ################
    #
    #   Creating the causal graphs
    #
    ################

    node_names = dict()
    for var in var_names:
        for lag in range(ground_truth_lags + 1):
            node_names[(var, lag)] = "L" + str(lag) + "." + str(var)

    # graph with only parents
    ground_truth_graph = nx.DiGraph()
    for key in node_names:
        ground_truth_graph.add_node(node_names[key])
    for key in ground_truth_parents:
        child_name = "L0." + str(key)
        for parent in ground_truth_parents[key]:
            parent_name = "L" + str(parent[1]) + "." + str(parent[0])
            ground_truth_graph.add_edge(parent_name, child_name)

    # summary graph (no lags)
    summary_graph = nx.DiGraph()
    summary_graph.add_nodes_from(var_names)
    for cause, effect in ground_truth_graph.edges:
        lag = cause[1:cause.find(".")]
        cause = cause[cause.find(".") + 1:]
        effect = effect[effect.find(".") + 1:]
        if not summary_graph.has_edge(cause, effect):
            summary_graph.add_edge(cause, effect, lags=[lag])
        else:
            summary_graph[cause][effect]["lags"].append(lag)

    # extractor functions

    def get_all_parents(graph, target):
        if "L0." + target in graph.nodes:
            return list(graph.predecessors("L0." + target))
        else:
            return list(graph.predecessors(target))

    def get_all_ancestors(graph, target):
        if "L0." + target in graph.nodes:
            return list(nx.ancestors(graph, "L0." + target))
        else:
            return list(nx.ancestors(graph, target))

    def get_all_connected(graph, target):
        if "L0." + target in graph.nodes:
            return list(nx.node_connected_component(graph.to_undirected(), "L0." + target))
        else:
            return list(nx.node_connected_component(graph.to_undirected(), target))

    causes_attributes_dict = defaultdict(list)

    for target_name in var_names:
        if cause_extraction == "parents":
            causes_attributes = get_all_parents(summary_graph, target_name)
        elif cause_extraction == "ancestors":
            causes_attributes = get_all_ancestors(summary_graph, target_name)
        elif cause_extraction == "connected":
            causes_attributes = get_all_connected(summary_graph, target_name)
        else:
            raise Exception("causeExtraction method specified in config file is not implemented")

        causes_attributes_dict[target_name] = list(map(str,causes_attributes))

    return df, var_names, causes_attributes_dict



