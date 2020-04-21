import pandas as pd
import matplotlib.pyplot as plt



#import sklearn packages
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# %% TRAINING WITH GRIDSEARCH CV REGRESSION


#train all linear regression models
def train_models_regression(X,y,models_regression,test_size=0.33,cv=5):

    # split into train and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # define evaluation dataframe
    D_trainingResults=pd.DataFrame(columns=['MODEL_NAME','MODEL','PARAMS','SCORE_TEST','SCORE_VALIDATION'])
    
    scores = 'neg_mean_squared_error'
    for model in models_regression.keys():
        estimator = models_regression[model]['estimator']
        param_grid = models_regression[model]['param']
        
        clf = GridSearchCV(estimator, param_grid,cv=cv,scoring=scores)
        clf.fit(X_train, y_train)
        #clf.cv_results_
        MODEL=clf.best_estimator_
        SCORE_TEST=clf.best_score_
        PARAMS=clf.best_params_
        
        #validation_set
        y_pred=MODEL.predict(X_test)
        #scorer=get_scorer(scores)
        SCORE_VALIDATION=-mean_squared_error(y_test, y_pred)
        D_trainingResults=D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, SCORE_VALIDATION ]],columns = D_trainingResults.columns))
    return D_trainingResults

# %% TRAINING WITH GRIDSEARCH CV REGRESSION
#train all linear classification models

def train_models_classification(X,y,models_classification,test_size=0.33,cv=5): 
    # split into train and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # define evaluation dataframe
    D_trainingResults=pd.DataFrame(columns=['MODEL_NAME','MODEL','PARAMS','SCORE_TEST','SCORE_VALIDATION'])
    
    scores = 'accuracy'
    for model in models_classification.keys():
        estimator = models_classification[model]['estimator']
        param_grid = models_classification[model]['param']
        
        clf = GridSearchCV(estimator, param_grid,cv=cv,scoring=scores)
        clf.fit(X_train, y_train)
        #clf.cv_results_
        MODEL=clf.best_estimator_
        SCORE_TEST=clf.best_score_
        PARAMS=clf.best_params_
        
        #validation_set
        y_pred=MODEL.predict(X_test)
        #scorer=get_scorer(scores)
        SCORE_VALIDATION=accuracy_score(y_test, y_pred)
        D_trainingResults=D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, SCORE_VALIDATION ]],columns = D_trainingResults.columns))
    return D_trainingResults


# %% DEBUG AREA REGRESSION

import warnings
warnings.filterwarnings("ignore")

#set path
import sys
import numpy as np
root_folder="C:\\Users\\aletu\\Documents\\GitHub\\OTHER\\ZENON"
sys.path.append(root_folder)

'''
# import models
# add linear models
from logproj.M_learningMethod.linear_models import models_regression as lin_class_mod

#add bayesian models
#from logproj.M_learningMethod.bayesians_models import models_regression as bay_class_mod

#add symbolist models
from logproj.M_learningMethod.symbolists_models import models_regression as sym_class_mod

#add connectionist models
from logproj.M_learningMethod.connectionists_models import models_regression as con_class_mod

#add analogizers models
from logproj.M_learningMethod.analogizers_models import models_regression as ana_class_mod

#add ensemble methods
from logproj.M_learningMethod.ensemble_methods import models_regression as ens_class_mod

all_models_regression={}
all_models_regression.update(lin_class_mod)
#all_models_regression.update(bay_class_mod)
all_models_regression.update(sym_class_mod)
all_models_regression.update(con_class_mod)
all_models_regression.update(ana_class_mod)
all_models_regression.update(ens_class_mod)

#import data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1]])
y = X[:, 0] ^ X[:, 1]
D_res_regr = train_models_regression(X,y,all_models_regression)

'''

# %% DEBUG AREA CLASSIFICATION
'''
# import models
all_models_classification={}

# add linear models
from logproj.M_learningMethod.linear_models import models_classification as lin_class_mod

#add bayesian models
from logproj.M_learningMethod.bayesians_models import models_classification as bay_class_mod

#add symbolist models
from logproj.M_learningMethod.symbolists_models import models_classification as sym_class_mod

#add connectionist models
from logproj.M_learningMethod.connectionists_models import models_classification as con_class_mod

#add analogizers models
from logproj.M_learningMethod.analogizers_models import models_classification as ana_class_mod

#add ensemble methods
from logproj.M_learningMethod.ensemble_methods import models_classification as ens_class_mod

all_models_classification.update(lin_class_mod)
all_models_classification.update(bay_class_mod)
all_models_classification.update(sym_class_mod)
all_models_classification.update(con_class_mod)
all_models_classification.update(ana_class_mod)
all_models_classification.update(ens_class_mod)

#import data

from sklearn import datasets
digits = datasets.load_digits()

X = digits.data[:-1]
y = digits.target[:-1]
D_res_class = train_models_classification(X,y,all_models_classification)


# %% CONFUSION MATRIX


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
'''
# In[1]: #confusion matrix


def plot_confusion_matrix_fromAvecm(ave_cm, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix from an average-precomputed confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = ave_cm
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
