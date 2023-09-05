


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
        self._prebuild_association_objects()
        
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
        if current_model.has_too_many_parameters():
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

        if self.ground_truth is not None:
            oracle_order = [var for var in self.ground_truth]
            if self.target in oracle_order:
                oracle_order.remove(self.target)
        algo_should_have_stopped = False
        
        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables)
            
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
            if self.ground_truth is not None:
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
        if len(self.selected_features) < self.config["max_features"] or current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
            if len(self.selected_features)>1:
                self.selected_features = self.selected_features[:-1]

        # flag class as being fitted
        self.fitted = True


