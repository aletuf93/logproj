from sklearn import svm

# %% GRID PARAMETERS CLASSIFICATION

tuned_param_svm= [{'kernel': ['rbf', 'linear'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],                
                  }]
                                
models_classification = {
                            'svm': {
                                   'estimator': svm.SVC(), 
                                   'param': tuned_param_svm,
                            },
                        }
# GRID PARAMETERS REGRESSION
# %% GRID PARAMETERS

tuned_param_regr= [{'kernel': ['rbf', 'linear'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],                
                  }]
                                
models_regression = {
                            'svm': {
                                   'estimator': svm.SVR(), 
                                   'param': tuned_param_regr,
                            },
                        }



