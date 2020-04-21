from sklearn import tree

# %% CLASSIFICATION MODEL PARAMETERS


tuned_param_dt= [{'criterion':['gini','entropy'],
                  'splitter':['best','random'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


                                
models_classification = {
                            'decision tree': {
                                   'estimator': tree.DecisionTreeClassifier(), 
                                   'param': tuned_param_dt,
                            },
                            
                           
                        }



# %% REGRESSION MODEL PARAMETERS

tuned_param_dt_regr= [{'criterion':['mse','mae'],
                  'splitter':['best','random'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


                                
models_regression = {
                            'decision tree': {
                                   'estimator': tree.DecisionTreeRegressor(), 
                                   'param': tuned_param_dt_regr,
                            },
                            
                           
                        }

# %% OLD CLASSIFICATION MODELS
'''
def decisionTreeCV(X,y,dirResults,nFolds,saveFig):
    
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
            model = DecisionTreeClassifier().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            score=np.round(metrics.roc_auc_score(y_test, y_pred),3)   
            resultScore.append(score)
            cm.append(metrics.confusion_matrix(y_test, y_pred))
            
    #plot
    if saveFig:
        ave_cm=np.mean(cm,axis=0)
        fig=plot_confusion_matrix_fromAvecm(ave_cm,['True','False'], title='Decision Tree')
        fig.savefig(dirResults+'\\confusionMatrixDecisionTree.png')
                
            
            
    mean_AUC=np.mean(resultScore)
    std_AUC=np.std(resultScore)
    plt.close('all')
    return mean_AUC, std_AUC, fig1 
'''

'''
def regressionTreeCV(X,y,nFolds, saveFig):
    
    #si puà aggiungere il max_depth="" dell'albero
    
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
        model = tree.DecisionTreeRegressor().fit(X_train, y_train)
    
        #Testo modello sul test set
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
        resultMSE.append(mse)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Regression Tree')
            plt.axis('equal')
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''
