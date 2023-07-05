from statsmodels.tsa.vector_ar.var_model import VAR





def bivariate_granger(data, target, maxlags, signif=0.05):
    selected = []
    for column in data.columns:
        if column==target:
            continue
        model = VAR(data[[target,column]])
        results = model.fit(maxlags=maxlags)
        pvalue = results.test_causality(target, causing=column, signif=signif).pvalue
        if pvalue < signif:
            pvalue = results.test_causality(column, causing=target, signif=signif).pvalue
            if pvalue > signif:
                selected.append(column)
    return selected
    

