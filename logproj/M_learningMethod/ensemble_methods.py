from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
# %% CLASSIFICATION MODEL PARAMETERS

tuned_param_rf= [{'n_estimators':[10,50,100,200],
                  #'criterion':['gini','entropy'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


tuned_param_ab = [{'n_estimators':[1, 10, 50, 100]               
                  }]


tuned_param_gb =  [{'loss':['deviance','exponential'],
                    'learning_rate':[1e-2, 1e-1, 1, 10],
                    'n_estimators':[1,2,10,50,100]
                  
                              
                  }]

tuned_param_bt = [{'n_estimators':[1, 10, 50, 100]               
                         }]
                                
models_classification = {
                            
                            
                            'random forest': {
                                   'estimator': RandomForestClassifier(), 
                                   'param': tuned_param_rf,
                            },
                            
                            'adaboost': {
                                   'estimator': AdaBoostClassifier(), 
                                   'param': tuned_param_ab,
                            },
                            
                            'gradient boosting': {
                                   'estimator': GradientBoostingClassifier(), 
                                   'param': tuned_param_gb,
                            },
                            
                            'bagging tree': {
                                   'estimator': BaggingClassifier(), 
                                   'param': tuned_param_bt,
                            },
                        }

# %% REGRESSION MODEL PARAMETERS

tuned_param_rf_regr= [{'n_estimators':[10,50,100,200],
                            #'criterion':['gini','entropy'],
                          'max_features':['auto','sqrt','log2'],
                          'max_depth': range(1,20)                 
                          }]

tuned_param_ab_regr  = [{'n_estimators':[1, 10, 50, 100]               
                         }]

tuned_param_gb_regr = [{'loss':['ls','lad','huber','quantile'],
                    'learning_rate':[1e-2, 1e-1, 1, 10],
                    'n_estimators':[1,2,10,50,100]
                  
                              
                      }]

tuned_param_bt_regr = [{'n_estimators':[1, 10, 50, 100]               
                         }]


models_regression = {
                            
                            
                            'random forest': {
                                   'estimator': RandomForestRegressor(), 
                                   'param': tuned_param_rf_regr,
                            },
                            
                            'adaboost': {
                                   'estimator': AdaBoostRegressor(), 
                                   'param': tuned_param_ab_regr,
                            },
                            
                            'gradient boosting': {
                                   'estimator': GradientBoostingRegressor(), 
                                   'param': tuned_param_gb_regr,
                            },
                            
                            'bagging tree': {
                                   'estimator': BaggingRegressor(), 
                                   'param': tuned_param_bt_regr,
                            },
                            
                            
                        }



# %% OLD ENSEMBLE METHODS

'''
def baggingTreeCV(X,y,nFolds, saveFig):

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
            model = BaggingClassifier().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
            resultMSE.append(mse)
            
            #plot
            if saveFig:
                plt.scatter(y_test, y_pred)
                plt.xlabel('actual value')
                plt.ylabel('predicted value')
                plt.title('Bagging Tree')
                plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''


'''
def randomForestCV(X,y,nFolds, saveFig):
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
            model = RandomForestRegressor(bootstrap=True,n_estimators=100,max_features='sqrt',criterion='mse').fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
            resultMSE.append(mse)
            
            #plot
            if saveFig:
                plt.scatter(y_test, y_pred)
                plt.xlabel('actual value')
                plt.ylabel('predicted value')
                plt.title('Random Forest')
                plt.axis('equal')
            
            
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''



'''

def gradientBoostingCV(X,y,nFolds, saveFig):
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
            model = GradientBoostingRegressor().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
            resultMSE.append(mse)
            
            #plot
            if saveFig:
                plt.scatter(y_test, y_pred)
                plt.xlabel('actual value')
                plt.ylabel('predicted value')
                plt.title('Gradient Boosting')
                plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''


'''
def adaboostCV(X,y,dirResults,nFolds,saveFig):
    
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
            model = AdaBoostClassifier().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            score=np.round(metrics.roc_auc_score(y_test, y_pred),3)   
            resultScore.append(score)
            cm.append(metrics.confusion_matrix(y_test, y_pred))
            
    #plot
    if saveFig:
        ave_cm=np.mean(cm,axis=0)
        fig=plot_confusion_matrix_fromAvecm(ave_cm,['True','False'], title='Adaboost')
        fig.savefig(dirResults+'\\confusionMatrixAdaboost.png')
                
            
            
    mean_AUC=np.mean(resultScore)
    std_AUC=np.std(resultScore)
    plt.close('all')
    return mean_AUC, std_AUC, fig1 
'''

'''   
def randomForestCV(X,y,dirResults,nFolds,saveFig):
    
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
            model = RandomForestClassifier().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            score=np.round(metrics.roc_auc_score(y_test, y_pred),3)   
            resultScore.append(score)
            cm.append(metrics.confusion_matrix(y_test, y_pred))
            
    #plot
    if saveFig:
        ave_cm=np.mean(cm,axis=0)
        fig=plot_confusion_matrix_fromAvecm(ave_cm,['True','False'], title='Random Forest')
        fig.savefig(dirResults+'\\confusionMatrixRandomForest.png')
                
            
            
    mean_AUC=np.mean(resultScore)
    std_AUC=np.std(resultScore)
    plt.close('all')
    return mean_AUC, std_AUC, fig1 
'''

'''
def gradientBoostingCV(X,y,dirResults,nFolds,saveFig):
    
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
            model = GradientBoostingClassifier().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            score=np.round(metrics.roc_auc_score(y_test, y_pred),3)   
            resultScore.append(score)
            cm.append(metrics.confusion_matrix(y_test, y_pred))
            
    #plot
    if saveFig:
        ave_cm=np.mean(cm,axis=0)
        fig=plot_confusion_matrix_fromAvecm(ave_cm,['True','False'], title='Gradient Boosting')
        fig.savefig(dirResults+'\\confusionMatrixGradientBoosting.png')
                
            
            
    mean_AUC=np.mean(resultScore)
    std_AUC=np.std(resultScore)
    plt.close('all')
    return mean_AUC, std_AUC, fig1 
'''


'''
def adaBoostComplete(X,y):
    X=ZO_ml.dummyColumns(X)
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300).fit(X, y)
    return model

def adaBoostCV(X,y,nFolds, saveFig):
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
        model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=300).fit(X_train, y_train)
    
        #Testo modello sul test set
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)  
        resultMSE.append(mse)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('AdaBoost')
            plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''