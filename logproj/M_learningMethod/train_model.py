import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#import sklearn packages
from sklearn.metrics import mean_squared_error, accuracy_score
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
