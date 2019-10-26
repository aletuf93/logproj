#import py_compile
#py_compile.compile('ZO_ML_RegressionTreeModels.py')

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pacchetti machine learning
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, GradientBoostingRegressor, RandomForestRegressor
#from pyearth import Earth #MARS


#pacchetti statistici
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

#pacchetti ZO
import ml_machine_learning as ZO_ml




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

def MarsCV(X,y,nFolds,saveFig):
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
            model = Earth().fit(X_train, y_train)
        
            #Testo modello sul test set
            y_pred = model.predict(X_test)
            mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
            resultMSE.append(mse)
            
            #plot
            if saveFig:
                plt.scatter(y_test, y_pred)
                plt.xlabel('actual value')
                plt.ylabel('predicted value')
                plt.title('MARS')
                plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1


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


def compareModelsCV(X,y,dirResultsMachineLearning,nFolds, models, saveFig):
    modelsComparisoncols=['Model','MSE mean','MSE std']
    D_R=pd.DataFrame(columns=modelsComparisoncols)
    
    #regression tree
    if 'regression Tree' in models:
        mse_mean,mse_std, fig =regressionTreeCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Regression Tree', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsRegressionTree.png')
    
    #bagging tree
    if 'Bagging regression Tree' in models:
        mse_mean,mse_std,fig=baggingTreeCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Bagging regression Tree', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsBaggingTree.png')
    
    #MARS 
    if 'mars' in models:
        mse_mean,mse_std,fig=MarsCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Multivariate Adaptive Regression Spline', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsMARS.png')
    
    #gradient boosting
    if 'Gradient boosting' in models:
        mse_mean,mse_std, fig=gradientBoostingCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Gradient boosting', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsGradientBoosting.png')
    
    #randomForest
    if 'RandomForest' in models:
        mse_mean,mse_std, fig=randomForestCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['RandomForest', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsRandomForest.png')
    
    plt.close('all')
    
    return D_R
