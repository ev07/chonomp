

import copy
import matplotlib.pyplot as plt
import time
import numpy as np

"""
Structure of code:
 - tsGOMP class that implement the gammaOMP for time series main loop and hyperparameters
   - fit method that launches the whole algorithm while keeping the summary stats of the whole algo
   - transform method that transform the whole dataset to exclude nonselected columns
     - fit_transform
   - get_selected_features that return the name or index of the selected features
   - config attribute containing the model constructor and hyperparameters
 - Model class that should cover the Resid, Train methods of the pseudocode
   - fit method
   - significance method (log likelyhood or bic or R² depending on implemented)
   - residuals method (standard residual in continuous, probability residual when ordinal, matrix of logits residuals in
     multiclass)
 - Correlation class
   - fit method returning a score.
"""


##
#
#   Main class
#
##



class tsGOMP_AutoRegressive:
    def __init__(self, config, target, verbosity=0):
        """
        In this version, the first residuals are computed on an autoregressive model.
        Thus, the target variable is by default in the selected set.
        
        config contains:
         - method: define how the stopping criterion will be computed if applicable.
         - significance_threshold: define the difference in model significance to stop.
         - max_features: the maximal number of selected features
         - association: a list of constructors of associations, one for each variable.
         - association.config: parameters to give to each association constructor
         - model: the model constructor
         - model.config: the parameters to give to the model constructor.
        
        verbosity set at 1 keeps track of the models used.
        """
        self.target = target
        self.fitted = False
        self.selected_features = None
        self.config = config
        self.verbosity = verbosity
        self.history = []
        self.association_objects = None
        self.check_config()
        self._prebuild_association_objects()
    
    def check_config(self):
        assert "association" in self.config
        assert "association.config" in self.config
        assert "max_features" in self.config
        assert "significance_threshold" in self.config
        assert "method" in self.config
        assert "model" in self.config
        assert "model.config" in self.config
    
    def _prebuild_association_objects(self):
        "Initialize association objects for each variable in the dataset, to avoid doing it later"
        self.association_objects = dict()
        for variable in self.config["association"]:
            association_constructor = self.config["association"][variable]
            association_object = association_constructor(self.config["association.config"][variable])
            self.association_objects[variable] = association_object
    
    def _stopping_criterion(self, current_model, previous_model):
        """return True if we should continue with more variables, False to stop"""
        # if enough features were selected, stop
        if len(self.selected_features) >= self.config["max_features"]:
            return False
        # if this is the first iteration, continue
        if len(self.selected_features) == 1:
            return True

        threshold = self.config["significance_threshold"]
        metric = current_model.stopping_metric(previous_model, self.config["method"])
        return metric < threshold

    def _initialize_model(self, data):
        """Creates an autoregressive model on the target variable."""
        current_model = self.config["model"](self.config["model.config"], target=self.target)
        current_model.fit(data[[self.target]])

        return current_model

    def fit(self, data):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # initialization of selected and candidate variables, starting from an autoregressive model
        candidate_variables = set(data.columns)
        candidate_variables.remove(self.target)
        self.selected_features = [self.target]

        previous_model = None  # will be defined during the first iteration
        current_model = self._initialize_model(data)
        residuals = current_model.residuals()

        if self.verbosity:
            self.history.append(current_model)
        
        while self._stopping_criterion(current_model, previous_model):
            # find maximally associative variable to current residuals
            measured_associations = []
            for variable in candidate_variables:
                variable_association = self.association_objects[variable].association(residuals, data[[variable]])
                measured_associations.append((variable_association, variable))
            chosen_variable = max(measured_associations)[1]
            # put the chosen variable in the selected feature set
            self.selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            # compute new model with this variable
            previous_model = current_model
            current_model = self.config["model"](self.config["model.config"], target=self.target)
            current_model.fit(data[self.selected_features])
            # change residuals
            residuals = current_model.residuals()
            
            if self.verbosity:
                self.history.append(current_model)
        
        # remove the last selected feature if irrelevant:
        if len(self.selected_features) < self.config["max_features"] or current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
            self.selected_features = self.selected_features[:-1]

        # flag class as being fitted
        self.fitted = True
        
    def transform(self, data):
        if self.fitted:
            return data[self.selected_features]
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def get_selected_features(self):
        if self.fitted:
            return self.selected_features


#####
#
#   PLOT HELPERS
#
#####



    def print_model_history(self):
        for number, model in enumerate(self.history):
            print("Model", number, "over variables", *list(model.data.columns))
            print("    Model significance:", model.significance())
            print("    Model assumptions p-values:\n", model.verify_assumption())
            print("=="*10)

    def plot_model_significance(self):
        # plot aic and sse on a graph, to show how both criterion progress
        aic = []
        sse = []
        for model in self.history:
            aic.append(-model.significance())
            sse.append(model.sse())
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(1, len(aic)), aic[1:], label="aic", c="r")
        ax2.plot(sse, label="sse", c="b")
        fig.legend()
        ax1.set_ylabel("aic")
        ax2.set_ylabel("sse")
        ax1.set_xlabel("Number of extra variables in the model")
        plt.title("Measure of the model performance when the selected variable set grows")
        plt.savefig("tmp.png")
        plt.show()

    def plot_max_association_by_step(self):
        # plot the highest correlation found
        association = []
        for index, model in enumerate(self.history[:-1]):
            residuals = model.residuals()
            variable = self.get_selected_features()[index+1]
            datavariable = self.history[index+1].data[[variable]]
            association_constructor = self.config["association"][variable]
            association_object = association_constructor(self.config["association.config"][variable])
            variable_association = association_object.association(residuals, datavariable)
            association.append(variable_association)

        fig, ax1 = plt.subplots()
        ax1.plot(association, c="r")
        ax1.set_ylabel("Association intensity:"+self.config["association.config"][self.target]["return_type"])
        ax1.set_xlabel("Number of extra variables in the model")
        plt.title("Measure of the highest correlation with the residuals when the selected variable set grows")
        plt.savefig("tmp.png")
        plt.show()

    def plot_association_distribution_by_step(self, data):
        for index, model in enumerate(self.history[:-1]):
            association = []
            residuals = model.residuals()
            for variable in data.columns:
                if variable in self.get_selected_features()[:index+2]:
                    continue
                datavariable = data[[variable]]
                association_constructor = self.config["association"][variable]
                association_object = association_constructor(self.config["association.config"][variable])
                variable_association = association_object.association(residuals, datavariable)
                association.append(variable_association)

            fig, ax1 = plt.subplots()
            ax1.hist(association)
            ax1.set_ylabel("Association intensity:"+self.config["association.config"][self.target]["return_type"])
            ax1.set_xlabel("Number of extra variables in the model")

        plt.title("Distribution of the association with the residuals of successive models")
        plt.savefig("tmp.png")
        plt.show()





class tsGOMP_oracle(tsGOMP_AutoRegressive):
    def __init__(self, config, target, ground_truth, verbosity=0):
        """
        In this version, the first residuals are computed on an autoregressive model.
        Thus, the target variable is by default in the selected set.
        
        config contains:
         - method: define how the stopping criterion will be computed if applicable.
         - significance_threshold: define the difference in model significance to stop.
         - max_features: the maximal number of selected features
         - association: a list of constructors of associations, one for each variable.
         - association.config: parameters to give to each association constructor
         - model: the model constructor
         - model.config: the parameters to give to the model constructor.
        
        verbosity set at 1 keeps track of the models used.
        """
        super().__init__(config,target,verbosity)
        self.ground_truth = ground_truth
    
    def _stopping_criterion(self, current_model, previous_model, len_selected_features):
        """return True if we should continue with more variables, False to stop"""
        # if enough features were selected, stop
        if len_selected_features >= self.config["max_features"]:
            return False
        # if this is the first iteration, continue
        if len_selected_features == 1:
            return True
        
        threshold = self.config["significance_threshold"]
        metric = current_model.stopping_metric(previous_model, self.config["method"])
        return metric < threshold
        

    def fit(self, data):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # initialization of selected and candidate variables, starting from an autoregressive model
        candidate_variables = set(data.columns)
        candidate_variables.remove(self.target)
        selected_features = [self.target]

        previous_model = None  # will be defined during the first iteration
        time_modeltrain_start = time.time()
        current_model = self._initialize_model(data)
        time_modeltrain_end = time.time()
        
        residuals = current_model.residuals()

        oracle_order = [var for var in self.ground_truth]
        if self.target in oracle_order:
            oracle_order.remove(self.target)
        algo_should_have_stopped = False
        
        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            
            time_association_start = time.time()
            measured_associations = []
            for variable in candidate_variables:
                variable_association = self.association_objects[variable].association(residuals, data[[variable]])
                measured_associations.append((variable_association, variable))
            time_association_end = time.time()
            
            chosen_variable = max(measured_associations)[1]
            
            # for logging, find out which variables have P-values that are TP, TN, FP, FN.
            TP = sum(1 if -pval<0.05 and var in self.ground_truth else 0 for pval,var in measured_associations)
            FP = sum(1 if -pval<0.05 and var not in self.ground_truth else 0 for pval,var in measured_associations)
            FN = sum(1 if -pval>=0.05 and var in self.ground_truth else 0 for pval,var in measured_associations)
            TN = sum(1 if -pval>=0.05 and var not in self.ground_truth else 0 for pval,var in measured_associations)
            
            # history
            new_history_row = {"step": len(selected_features),
            "model": current_model,
            "associations": measured_associations,
            "associations_TP": TP,
            "associations_FP": FP,
            "associations_TN": TN,
            "associations_FN": FN,
            "associations_time": time_association_end - time_association_start,
            "association_chosen":chosen_variable,
            "chosen_in_ground_truth": chosen_variable in self.ground_truth,
            "should_have_stopped": algo_should_have_stopped,
            "remaining_causal": len(oracle_order),
            "previous_model_time": time_modeltrain_end - time_modeltrain_start
            }
            
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            if chosen_variable in oracle_order:
                oracle_order.remove(chosen_variable)
                
                
            # compute new model with this variable
            previous_model = current_model
            time_modeltrain_start = time.time()
            current_model = self.config["model"](self.config["model.config"], target=self.target)
            current_model.fit(data[selected_features])
            time_modeltrain_end = time.time()
            # change residuals
            residuals = current_model.residuals()

            # history
            new_history_row["stopping_metric"] = current_model.stopping_metric(previous_model, self.config["method"])
            
            
            #selected feature set of the natural algorithm
            if not algo_should_have_stopped and new_history_row["stopping_metric"] >= self.config["significance_threshold"]:
                algo_should_have_stopped = True
                new_history_row["current_is_last_model"] = True
                # chosen variable from this step onward should not be included in statistics about chosen variables, since it was not realy chosen.
            else:
                new_history_row["current_is_last_model"] = False
                
            self.history.append(new_history_row)
            
        # for logging, add the association step
        time_association_start = time.time()
        measured_associations = []
        for variable in candidate_variables:
            variable_association = self.association_objects[variable].association(residuals, data[[variable]])
            measured_associations.append((variable_association, variable))
        time_association_end = time.time()
        
        TP = sum(1 if -pval<0.05 and var in self.ground_truth else 0 for pval,var in measured_associations)
        FP = sum(1 if -pval<0.05 and var not in self.ground_truth else 0 for pval,var in measured_associations)
        FN = sum(1 if -pval>=0.05 and var in self.ground_truth else 0 for pval,var in measured_associations)
        TN = sum(1 if -pval>=0.05 and var not in self.ground_truth else 0 for pval,var in measured_associations)
                
        new_history_row = {"step": len(selected_features),
            "model":current_model,
            "associations": measured_associations,
            "associations_TP": TP,
            "associations_FP": FP,
            "associations_TN": TN,
            "associations_FN": FN,
            "associations_time": time_association_end - time_association_start,
            "association_chosen":None, 
            "chosen_in_ground_truth": None,
            "remaining_causal": None,
            "previous_model_time": time_modeltrain_end - time_modeltrain_start,
            "should_have_stopped": algo_should_have_stopped,
            "current_is_last_model": not algo_should_have_stopped, #if true, then chosen variable from this step onward should not be included.
            "stopping_metric": None}
            
        self.history.append(new_history_row)
        
        
        # create selected feature set
        
        self.selected_features = selected_features
        # remove the last selected feature if irrelevant:
        if len(self.selected_features) < self.config["max_features"] or current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
            self.selected_features = self.selected_features[:-1]

        # flag class as being fitted
        self.fitted = True
        






#############
#
#   Implementation for the special case of a single Association class that compute associations altogether.
#
#############




class tsGOMP_OneAssociation(tsGOMP_AutoRegressive):
    def __init__(self, config, target, ground_truth=None, verbosity=0):
        """
        In this version, the first residuals are computed on an autoregressive model.
        Thus, the target variable is by default in the selected set.
        
        config contains:
         - method: define how the stopping criterion will be computed if applicable.
         - significance_threshold: define the difference in model significance to stop.
         - max_features: the maximal number of selected features
         - association: the class constructor of the multivariate association to use
         - association.config: parameters to give to the association constructor
         - model: the model constructor
         - model.config: the parameters to give to the model constructor.
        
        verbosity set at 1 keeps track of the algorithm whole history.
        """
        super().__init__(config,target,verbosity)
        self.ground_truth = ground_truth
    
    def check_config(self):
        assert "association" in self.config
        assert "association.config" in self.config
        assert "max_features" in self.config
        assert "valid_obs_param_ratio" in self.config
        assert "significance_threshold" in self.config
        assert "method" in self.config
        assert "model" in self.config
        assert "model.config" in self.config
    
    def _prebuild_association_objects(self):
        "Initialize association object to respect the main algorithm structure"
        association_constructor = self.config["association"]
        association_object = association_constructor(self.config["association.config"])
        self.association_objects = association_object
    
    def _stopping_criterion(self, current_model, previous_model, len_selected_features):
        """return True if we should continue with more variables, False to stop"""
        # if enough features were selected, stop
        if len_selected_features >= self.config["max_features"]:
            return False
        # if the number of observations is too low compared to the number of parameters, stop
        if current_model.has_too_many_parameters(self.config["valid_obs_param_ratio"]):
            return False
        # if this is the first iteration, continue
        if len_selected_features == 1:
            return True
        
        threshold = self.config["significance_threshold"]
        metric = current_model.stopping_metric(previous_model, self.config["method"])
        return metric < threshold
        
    def _train_model(self, data, selected_set):
        """Creates a model on the selected variables."""
        current_model = self.config["model"](self.config["model.config"], target=self.target)
        current_model.fit(data[selected_set])

        return current_model
    
    def _initialize_fit(self, initial_selected, data):
    
        #prepare the initial regression set
        if initial_selected == []:
            initial_selected = [self.target]
        elif self.target not in initial_selected:
            initial_selected = [self.target] + initial_selected
    
        # initialization of candidate variables
        candidate_variables = set(data.columns)
        for variable in initial_selected:
            candidate_variables.remove(variable)
            
        # initialize the N-1 model so that the stopping criterion can be used from the first step
        if len(initial_selected)==1:
            previous_model = None  # will be defined during the first iteration
        else:  # train on the selected, removing the last included variable.
            previous_model = self._train_model(data, initial_selected[:-1])
        
        # initialize the current model
        time_modeltrain_start = time.time()
        current_model = self._train_model(data, initial_selected)
        time_modeltrain_end = time.time()
        
        residuals = current_model.residuals()
        
        return initial_selected, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end

    def _forward_verbose(self, data, initial_selected=[]):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # train first model with the list of covariate given.
        selected_features, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end = self._initialize_fit(initial_selected, data)

        #logging
        if self.ground_truth is not None:
            oracle_order = [var for var in self.ground_truth]
            if self.target in oracle_order:
                oracle_order.remove(self.target)
        algo_should_have_stopped = False
        
        
        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables) 
            if len(candidate_variable_list)==0:  # verify that we still have candidates
                break
            
            # compute associations
            time_association_start = time.time()
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            time_association_end = time.time()
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variable_list[chosen_index]
            
            
            if self.verbosity:
                # the following only works when the association outputs a p-value
                # for logging, find out which variables have P-values that are TP, TN, FP, FN.
                if self.ground_truth is not None:
                    TP = sum(1 if -pval<0.05 and var in self.ground_truth else 0 for pval,var in zip(measured_associations, candidate_variable_list))
                    FP = sum(1 if -pval<0.05 and var not in self.ground_truth else 0 for pval,var in zip(measured_associations, candidate_variable_list))
                    FN = sum(1 if -pval>=0.05 and var in self.ground_truth else 0 for pval,var in zip(measured_associations, candidate_variable_list))
                    TN = sum(1 if -pval>=0.05 and var not in self.ground_truth else 0 for pval,var in zip(measured_associations, candidate_variable_list))
                
                # history
                new_history_row = {"step": len(selected_features),
                "model": current_model,
                "associations": list(zip(measured_associations, candidate_variable_list)),
                "associations_time": time_association_end - time_association_start,
                "association_chosen":chosen_variable,
                "chosen_in_ground_truth": chosen_variable in self.ground_truth if self.ground_truth is not None else None,
                "should_have_stopped": algo_should_have_stopped,
                "remaining_causal": len(oracle_order) if self.ground_truth is not None else None,
                "previous_model_time": time_modeltrain_end - time_modeltrain_start
                }
                if self.ground_truth is not None:
                    new_history_row = {**new_history_row,
                                      "associations_TP": TP,
                                      "associations_FP": FP,
                                      "associations_TN": TN,
                                      "associations_FN": FN}
                                  
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            
            # modify ground truth if provided, to keep on computing causal stats.
            if self.ground_truth is not None:
                if chosen_variable in oracle_order:
                    oracle_order.remove(chosen_variable)
                
                
            # compute new model with this variable
            previous_model = current_model
            time_modeltrain_start = time.time()
            current_model = self._train_model(data, selected_features)
            time_modeltrain_end = time.time()
            # change residuals
            residuals = current_model.residuals()

            # history
            if self.verbosity:
                new_history_row["stopping_metric"] = current_model.stopping_metric(previous_model, self.config["method"])
                
                
                #selected feature set of the natural algorithm
                if not algo_should_have_stopped and new_history_row["stopping_metric"] >= self.config["significance_threshold"]:
                    algo_should_have_stopped = True
                    new_history_row["current_is_last_model"] = True
                    # chosen variable from this step onward should not be included in statistics about chosen variables, since it was not realy chosen.
                else:
                    new_history_row["current_is_last_model"] = False
                    
                self.history.append(new_history_row)
        
        if self.verbosity:
            # for logging, add the association step
            candidate_variable_list = list(candidate_variables)
            time_association_start = time.time()
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            time_association_end = time.time()
            
            # the following only works when the association outputs a p-value
            if self.ground_truth is not None:
                TP = sum(1 if -pval<0.05 and var in self.ground_truth else 0 for pval,var in zip(measured_associations,candidate_variable_list))
                FP = sum(1 if -pval<0.05 and var not in self.ground_truth else 0 for pval,var in zip(measured_associations,candidate_variable_list))
                FN = sum(1 if -pval>=0.05 and var in self.ground_truth else 0 for pval,var in zip(measured_associations,candidate_variable_list))
                TN = sum(1 if -pval>=0.05 and var not in self.ground_truth else 0 for pval,var in zip(measured_associations,candidate_variable_list))
                    
            new_history_row = {"step": len(selected_features),
                "model":current_model,
                "associations": list(zip(measured_associations,candidate_variable_list)),
                "associations_time": time_association_end - time_association_start,
                "association_chosen":None, 
                "chosen_in_ground_truth": None,
                "remaining_causal": None,
                "previous_model_time": time_modeltrain_end - time_modeltrain_start,
                "should_have_stopped": algo_should_have_stopped,
                "current_is_last_model": not algo_should_have_stopped, #if true, then chosen variable from this step onward should not be included.
                "stopping_metric": None}
            if self.ground_truth is not None:
                new_history_row = {**new_history_row,
                    "associations_TP": TP,
                    "associations_FP": FP,
                    "associations_TN": TN,
                    "associations_FN": FN}
            
            self.history.append(new_history_row)
        
        
        # create selected feature set
        
        self.selected_features = selected_features
        # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
        if len(self.selected_features)>1:
            if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                self.selected_features = self.selected_features[:-1]

    def _forward(self, data, initial_selected=[]):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # separated verbosity logging, for clarity.
        if self.verbosity:
            return self._forward_verbose(data, initial_selected)
        
        
        # train first model with the list of covariate given.
        selected_features, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end = self._initialize_fit(initial_selected, data)

        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables) 
            if len(candidate_variable_list)==0:  # verify that we still have candidates
                break
            
            # compute associations
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variable_list[chosen_index]
                                  
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            
            # compute new model with this variable
            previous_model = current_model
            current_model = self._train_model(data, selected_features)
            # change residuals
            residuals = current_model.residuals()

        # create selected feature set
        self.selected_features = selected_features
        # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
        if len(self.selected_features)>1:
            if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                self.selected_features = self.selected_features[:-1]
    
    
    
    def fit(self, data, initial_selected=[]):
        """
        Given a dataset, compute the forward pass over it.
        It is possible to give a set of initially selected columns, to start the algorithm forward phase by a non-empty provided set.
        
        Params:
            data: pd.DataFrame, containing the forecasted MTS. Must be ordered by timestamp increasing.
            initial_selected (optional): list of str, the list of columns to use as initial set of the forward phase. Default is empty.
        """
    
        self._forward(data, initial_selected)
        self.fitted = True 
        
    def fit_from_selected_list(self, data, selected_list):
        """
        Given a list of ordered column names, corresponding to a run of the algorithm with the same config except for the selection threshold,
        run the current configuration by using the list to avoid computing the correlations, shortening the process.
        
        Params:
            data: pd.DataFrame, the dataframe containing the forecasted MTS. Must be ordered by timestamp increasing.
            selected_list: list of str, ordered list of the covariates (columns in the dataframe),
                 that have been insered in order by a previous run of the algorithm with same parameters except the stopping thresholds
                 (significance_threshold, max_features, valid_obs_params_ratio).
        """
        #first step, runing each model and verifying that the stopping criterion does not stop the algorithm before end of given sequence
        
        # initial model
        selected_features, _, previous_model, current_model, _, _, _ = self._initialize_fit([], data)
        if self.target == selected_list[0]:
            selected_list = selected_list[1:]
        elif self.target in selected_list:
            selected_list.remove(self.target)
        # loop over variables
        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            candidate_variable_list = list(candidate_variables) 
            if len(selected_list)==0:  # verify that we still have candidates in the history
                break
            next_chosen = selected_list.pop(0)
            selected_features.append(next_chosen)
            previous_model = current_model
            current_model = self._train_model(data, selected_features)
        # if termination due to stopping criterion, remove the eventual superflous 
        else:
            self.selected_features = selected_features
            # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
            if len(self.selected_features)>1:
                if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                    self.selected_features = self.selected_features[:-1]
            self.fitted = True
            return
        
        #second step, continuing with the standard algorithm for new selected variable if this point is reached.
        self._forward(data, initial_selected=selected_list)
        self.fitted = True
        
    def _backward(self, data):
        """
        Backward pass testing each of the covariate for nonzero coefficient.
        The test is Y_t indep X_t-L...X_t-1 | Z_t-L..Z_t-1 where Z are all other covariates including Y_t-L...Y_t-1
        While there is such a change, keep on conducting backward tests.
        
        Params:
            data: pd.DataFrame, containing the time series that will be forecasted. Must be ordered by timestamp increasing.
        """
        selected_set_has_changed = True
        while selected_set_has_changed:
            selected_set_has_changed = False  # flag reset
            selected_features = copy.copy(self.selected_features)
            full_model = self._train_model(data, selected_features)
            for column in [x for x in selected_features if x!=self.target]:  # never remove target
                restricted_model = self._train_model(data, [x for x in selected_features if x!=column])
                threshold = self.config["significance_threshold_backward"]
                metric = full_model.stopping_metric(restricted_model, self.config["method_backward"])
                if metric >= threshold:  # there is no significative difference between the models.
                    self.selected_features.remove(column)
                    selected_set_has_changed = True  # set change flag to true
    
    def fit_backward(self, data, initial_selected=[]):
        """
        Fit the forward-backward version of the algorithm.
        Params:
            data: pd.DataFrame, the MTS ordered by timestamp increasing.
        """
        self._forward(data, initial_selected=initial_selected)
        self._backward(data)
        self.fitted = True


class tsGOMP_train_val(tsGOMP_OneAssociation):
    def check_config(self):
        super().check_config()
        assert "validation_ratio" in self.config
    def _train_val_data_split(self, data):
        validation_size = int(self.config.get("validation_ratio", 0.1)*len(data))
        data_train = data.iloc[:-validation_size]
        data_test = data.iloc[-validation_size:]
        return data_train, data_test
    def _train_model(self, data, selected_set):
        """Creates a model on the selected variables.
        In this version, also uses a train val split to compute residuals."""
        current_model = self.config["model"](self.config["model.config"], target=self.target)
        data_train, data_test = self._train_val_data_split(data[selected_set])
        current_model.fit(data_train)
        r=current_model.residuals(data_test, test=True)
        return current_model

class tsGOMP_multiple_subsets(tsGOMP_OneAssociation):
    """
    My understanding of what is done in epilogi.
    
    For each added variable (which is non-redundant to the selected set), the remaining set is searched for equivalent redundant variables.
    Any redundant variable is added to the equivalency of the newly added variable.
    
    Then we can pick and choose any combination and replacement of the selected set.
    """
    def check_config(self):
        super().check_config()
        assert "partial_correlation" in self.config
        assert "partial_correlation.config" in self.config
        assert "equivalence_threshold" in self.config
    
    def _prebuild_association_objects(self):
        super()._prebuild_association_objects()
        # add partial correlation object. For now, unique.
        partial_corr_constructor = self.config["partial_correlation"]
        partial_correlation_object = association_constructor(self.config["partial_correlation.config"])
        self.partial_correlation_objects = partial_correlation_object
    
    def _equivalent_set(self, data, chosen_variable, residuals, candidate_variables):
        """
        Compute for each candidate variable, if it is equivalent to the chosen variable by partial correlation with residuals.
        """
        equivalence_threshold = self.config["equivalence_threshold"]
        equivalent_list = []
        for candidate in candidate_variables:
            candidate_df = data[[candidate]]
            residuals_df = residuals
            condition_df = data[[chosen_variable]]
            pvalue = self.partial_correlation_objects.partial_corr(residuals_df, candidate_df, condition_df)
            if pvalue < equivalence_threshold:
                pvalue = self.partial_correlation_objects.partial_corr(residuals_df, condition_df, candidate_df)
            

    def _forward(self, data, initial_selected=[]):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # train first model with the list of covariate given.
        selected_features, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end = self._initialize_fit(initial_selected, data)
        
        # keep track of the equivalent covariates to each covariate.
        equivalent_variables = dict()

        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables) 
            if len(candidate_variable_list)==0:  # verify that we still have candidates
                break
            
            # compute associations
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variable_list[chosen_index]
                                  
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            
            # compute equivalent set and remove equivalent features
            equivalent_variables[chosen_variable] = self._equivalent_set(data, chosen_variable, residuals, candidate_variables)
            for to_remove in equivalent_features[chosen_variable]:
                candidate_variables.remove(to_remove)
            
            # compute new model with this variable
            previous_model = current_model
            current_model = self._train_model(data, selected_features)
            # change residuals
            residuals = current_model.residuals()

        # create selected feature set
        self.selected_features = selected_features
        self.equivalent_variables = equivalent_variables
        # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
        if len(self.selected_features)>1:
            if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                self.selected_features = self.selected_features[:-1]
    
    
