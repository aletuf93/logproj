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
# %% OLD CLASSIFICATIION METHODS
'''
def supportVectorRegressionCV(X,y,nFolds,saveFig):
    X=ZO_ml.dummyColumns(X)
    kf = KFold(n_splits=nFolds)
    kf.get_n_splits(X)

    k=0
    resultMSE=[]
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
       
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        #Costruisco modello sul training set
        model = SVR().fit(X_train, y_train)
    
        #Testo modello sul test set
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)  
        resultMSE.append(mse)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Support Vector regression')
            plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''




'''
def supportVectorMachineCV(X,y,dirResults,nFolds,saveFig):
    
    X=ZO_ml.dummyColumns(X)
    kf = KFold(n_splits=nFolds)
    kf.get_n_splits(X)
    
    k=0
    resultScore=[]
    fig1=[]
    if saveFig:
        fig1= plt.figure()
    cm=[]
    
    for train_index, test_index in kf.split(X):
            k=k+1 #identifico il fold
           
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            #Costruisco modello sul training set
            model = SVC().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            score=np.round(metrics.roc_auc_score(y_test, y_pred),3)   
            resultScore.append(score)
            cm.append(metrics.confusion_matrix(y_test, y_pred))
            
    #plot
    if saveFig:
        ave_cm=np.mean(cm,axis=0)
        fig=plot_confusion_matrix_fromAvecm(ave_cm,['True','False'], title='SVM')
        fig.savefig(dirResults+'\\confusionMatrixSVM.png')
                
            
            
    mean_AUC=np.mean(resultScore)
    std_AUC=np.std(resultScore)
    plt.close('all')
    return mean_AUC, std_AUC, fig1 
'''


