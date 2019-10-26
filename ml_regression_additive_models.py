#import py_compile
#py_compile.compile('ZO_ML_RegressionAdditiveModels.py')

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#pacchetti machine learning
#from pygam import LinearGAM,LogisticGAM,PoissonGAM,GammaGAM
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#pacchetti statistici
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

#pacchetti ZO
import ml_machine_learning as ZO_ml



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
def GAMregressionComplete(X,y,model):
    X=ZO_ml.dummyColumns(X)
    
    if model=='linear':
        m=LinearGAM()
    elif model=='poisson':
        m=PoissonGAM()
    elif model=='gamma':
        m=GammaGAM()
    
    
    mm=m.fit(X, y)
    return mm

def GAMregressionCV(X,y,nFolds,model, saveFig):
    if model=='linear':
        m=LinearGAM()
    elif model=='poisson':
        m=PoissonGAM()
    elif model=='gamma':
        m=GammaGAM()
    

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
        model = m.fit(X_train, y_train)
    
        #Testo modello sul test set
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)   
        resultMSE.append(mse)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Generalized additive model: '+model)
            plt.axis('equal')
        
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1
'''

def perceptronNeuralNetworkCV(X,y,nFolds,saveFig):
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
        model = MLPRegressor().fit(X_train, y_train)
    
        #Testo modello sul test set
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)  
        resultMSE.append(mse)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Neural Network - single perceptron')
            plt.axis('equal')
            
    mean_MSE=np.mean(resultMSE)
    std_MSE=np.std(resultMSE)
    return mean_MSE, std_MSE, fig1


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


def compareModelsBootstrap(X,y,nBoot,dirResultsMachineLearning,nFolds):
    modelsComparisoncols=['Model','MSE mean','MSE std']
    D_R=pd.DataFrame(columns=modelsComparisoncols)
    
    #Generalized additive models (GAM)
    mms=['linear','logistic','poisson']
    for mm in mms:
        try:
            model=GAMregressionComplete(X,y,mm)
            D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
            temp=pd.DataFrame([[mm +' Additive Model', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
            D_R=D_R.append(temp) 
        except:
             print('error with: '+mm)
    
    
    #Adaptive Boosting (ADABOOST)
    model=adaBoostComplete(X,y)
    D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
    temp=pd.DataFrame([['AdaBoost regression', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
    D_R=D_R.append(temp)
    
    
    return D_R





def compareModelsCV(X,y,dirResultsMachineLearning,nFolds, models, saveFig):
    modelsComparisoncols=['Model','MSE mean','MSE std']
    D_R=pd.DataFrame(columns=modelsComparisoncols)
    
    if 'GAM' in models:
        #Generalized additive models (GAM)
        mms=['linear','poisson','gamma']
        for mm in mms:
            try:
                mse_mean,mse_std, fig =GAMregressionCV(X,y,nFolds,mm,saveFig)
                temp=pd.DataFrame([[mm+ ' Additive Model', mse_mean,mse_std]],columns=modelsComparisoncols)
                D_R=D_R.append(temp)
                
                if saveFig: fig.savefig(dirResultsMachineLearning+'\\predictionsAdditive'+mms+'.png')
            except:
                 print('error with: '+mm)
    
    #Adaptive Boosting (ADABOOST)
    if 'adaboost' in models:
        mse_mean,mse_std, fig =adaBoostCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['AdaBoost regression', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\adaboost.png')
    
    #Single layer neural network
    if 'neuralnet' in models:
        mse_mean,mse_std, fig =perceptronNeuralNetworkCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Neural network - single perceptron', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\neuralNet.png')
        
    if 'SVR' in models:
        supportVectorRegressionCV
        mse_mean,mse_std, fig =supportVectorRegressionCV(X,y,nFolds,saveFig)
        temp=pd.DataFrame([['Support Vector Regression', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if saveFig: fig.savefig(dirResultsMachineLearning+'\\svr.png')
    
    
    plt.close('all')
    
    return D_R

