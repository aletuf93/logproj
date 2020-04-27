#import py_compile
#py_compile.compile('ZO_ML_RegressionLinearModels.py')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#pacchetti statistici
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

#pacchetti machine learning
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, LassoCV, Lasso, ElasticNetCV, ElasticNet, LarsCV, Lars

#pacchetti ZO
#import ML_machineLearning as ZO_ml


#Linear regression fit
def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = metrics.mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared



    
#modello di regressione lineare sul set completo    
def ordinaryLeastSquareComplete(X,y):
    
   
    
    #Costruisco regressione lineare sul train-set
    regr = LinearRegression()
    lr=regr.fit(X, y)    
    return lr

#modello di regressione lineare con cross validation   
def ordinaryLeastSquareCV(X,y,dirResults,nFolds,saveFig):
    
   
             
    #costruisco tabella risultati
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=('MSE') #l'ultima colonna aveva la variabile target. La uso per il MSE
    results=pd.DataFrame(columns=columnNames)
    
    
    numFolds=nFolds
    
    kf = KFold(n_splits=numFolds)
    kf.get_n_splits(X)
    
    print(kf)  
    k=0
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
        
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #Costruisco regressione lineare sul train-set
        regr = LinearRegression()
        lr=regr.fit(X_train, y_train)
    
        #Testo la regressione lineare sul test-set
        y_pred = lr.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
        beta=np.round(regr.coef_,2)
        
        newRow=list(beta)
        newRow.append(mse)
        newRow=[newRow]
        row=pd.DataFrame(newRow,columns=columnNames)
        results=results.append(row)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Ordinary least square')
            plt.axis('equal')
            
    if saveFig: results.to_html(dirResults+'\\02-OLSResults_'+str(numFolds)+'fold_CV.html')
    mean_MSE=np.round(np.mean(results.MSE),2)
    std_MSE=np.round(np.std(results.MSE),2)
    
    
    return results, mean_MSE, std_MSE, fig1
 
# modello di Ridge regression con cross validation
def RidgeRegressionCV(X,y,dirResults,nFolds, saveFig):
    
        
    numFolds=nFolds
    alpha_ridge = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]
      
    #costruisco tabella risultati
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=('MSE') #l'ultima colonna aveva la variabile target. La uso per il MSE
    columnNames.append('lambda')
    results=pd.DataFrame(columns=columnNames)
    
    
    
    kf = KFold(n_splits=numFolds)
    kf.get_n_splits(X)
    
    print(kf)  
    k=0
    #Cerco il miglior valore di alpha (lambda)
    miglioriAlpha=[]
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
        
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #trovo il miglior alpha con CV
        modelCV = RidgeCV(alphas=alpha_ridge, normalize=True) #aggiungere scoring su MSE
        modelCV.fit(X_train, y_train)
        alphaValue=modelCV.alpha_
        miglioriAlpha.append(alphaValue)
        
    #scelgo il valore di alpha più ricorrente fra i vari train e test set 
    alphaValue=max(set(miglioriAlpha), key=miglioriAlpha.count)
    
    #applico la Ridge Regression con alpha trovato
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = Ridge(alpha=alphaValue, normalize=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
        beta=np.round(model.coef_,2)
        
        newRow=list(beta)
        newRow.append(mse)
        newRow.append(alphaValue)
        newRow=[newRow]
        row=pd.DataFrame(newRow,columns=columnNames)
        results=results.append(row)
        
        #plot
        if saveFig: 
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Ridge Regression')
            plt.axis('equal')
        
    if saveFig: results.to_html(dirResults+'\\03-RidgeResults_'+str(numFolds)+'fold_CV.html')
    mean_MSE=np.round(np.mean(results.MSE),2)
    std_MSE=np.round(np.std(results.MSE),2)
    return results, mean_MSE, std_MSE, alphaValue, fig1

def RidgeRegressionComplete(X,y,alphaValue):
    
   
    
    model = Ridge(alpha=alphaValue, normalize=True)
    mm=model.fit(X,y)
    return mm

 
def LassoRegressionCV(X,y,dirResults,nFolds, saveFig):
        
    numFolds=nFolds
    alpha_lasso = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]
      
    #costruisco tabella risultati
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=('MSE') #l'ultima colonna aveva la variabile target. La uso per il MSE
    columnNames.append('lambda')
    results=pd.DataFrame(columns=columnNames)
    
    
    
    kf = KFold(n_splits=numFolds)
    kf.get_n_splits(X)
    
    print(kf)  
    k=0
    #Cerco il miglior valore di alpha (lambda)
    miglioriAlpha=[]
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
        
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #trovo il miglior alpha con CV
        modelCV = LassoCV(alphas=alpha_lasso, normalize=True) #aggiungere scoring su MSE
        modelCV.fit(X_train, y_train)
        alphaValue=modelCV.alpha_
        miglioriAlpha.append(alphaValue)
        
    #scelgo il valore di alpha più ricorrente fra i vari train e test set 
    alphaValue=max(set(miglioriAlpha), key=miglioriAlpha.count)
    
    #applico la Ridge Regression con alpha trovato
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = Lasso(alpha=alphaValue, normalize=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
        beta=np.round(model.coef_,2)
        
        newRow=list(beta)
        newRow.append(mse)
        newRow.append(alphaValue)
        newRow=[newRow]
        row=pd.DataFrame(newRow,columns=columnNames)
        results=results.append(row)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Lasso Regression')
            plt.axis('equal')
        
    if saveFig: results.to_html(dirResults+'\\04-LassoResults_'+str(numFolds)+'fold_CV.html')
    mean_MSE=np.round(np.mean(results.MSE),2)
    std_MSE=np.round(np.std(results.MSE),2)
    return results, mean_MSE, std_MSE, alphaValue, fig1

def LassoRegressionComplete(X,y,alphaValue):
    
   
    
    model = Lasso(alpha=alphaValue, normalize=True)
    mm=model.fit(X,y)
    return mm

def ElasticNetRegressionCV(X,y,dirResults,nFolds, saveFig):
   
    
    
    numFolds=nFolds
    alpha_en = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]
      
    #costruisco tabella risultati
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=('MSE') #l'ultima colonna aveva la variabile target. La uso per il MSE
    columnNames.append('lambda')
    results=pd.DataFrame(columns=columnNames)
    
    
    
    kf = KFold(n_splits=numFolds)
    kf.get_n_splits(X)
    
    print(kf)  
    k=0
    #Cerco il miglior valore di alpha (lambda)
    miglioriAlpha=[]
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
        
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #trovo il miglior alpha con CV
        modelCV = ElasticNetCV(alphas=alpha_en, normalize=True) #aggiungere scoring su MSE
        modelCV.fit(X_train, y_train)
        alphaValue=modelCV.alpha_
        miglioriAlpha.append(alphaValue)
        
    #scelgo il valore di alpha più ricorrente fra i vari train e test set 
    alphaValue=max(set(miglioriAlpha), key=miglioriAlpha.count)
    
    #applico la Ridge Regression con alpha trovato
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = ElasticNet(alpha=alphaValue, normalize=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
        beta=np.round(model.coef_,2)
        
        newRow=list(beta)
        newRow.append(mse)
        newRow.append(alphaValue)
        newRow=[newRow]
        row=pd.DataFrame(newRow,columns=columnNames)
        results=results.append(row)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('Elastic Net Regression')
            plt.axis('equal')
        
    if saveFig: results.to_html(dirResults+'\\05-elasticNetResults_'+str(numFolds)+'fold_CV.html')
    mean_MSE=np.round(np.mean(results.MSE),2)
    std_MSE=np.round(np.std(results.MSE),2)
    return results, mean_MSE, std_MSE,alphaValue, fig1

def ElasticNetRegressionComplete(X,y,alphaValue):
    
    
    
    model = ElasticNet(alpha=alphaValue, normalize=True)
    mm=model.fit(X,y)
    return mm

def LARSregression(X,y,dirResults,nFolds, saveFig):
    
    numFolds=nFolds
    
    #se sono rimaste delle variabili categoriche nel tableau X le trasformo in dummy
    for column in X.columns:
     if X[column].dtype==object:
         dummyCols=pd.get_dummies(X[column])
         X=pd.concat([X,dummyCols], axis=1)
         del X[column]
      
    #costruisco tabella risultati
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=('MSE') #l'ultima colonna aveva la variabile target. La uso per il MSE
    results=pd.DataFrame(columns=columnNames)
    
    
    
    kf = KFold(n_splits=numFolds)
    kf.get_n_splits(X)
    
    print(kf)  
    k=0
    
    fig1=[]
    if saveFig:
        fig1=plt.figure()
    for train_index, test_index in kf.split(X):
        k=k+1 #identifico il fold
        
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        
        
        #applico LARS
        model = LarsCV()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
        beta=np.round(model.coef_,2)
        newRow=list(beta)
        newRow.append(mse)
        
        newRow=[newRow]
        row=pd.DataFrame(newRow,columns=columnNames)
        results=results.append(row)
        
        #plot
        if saveFig:
            plt.scatter(y_test, y_pred)
            plt.xlabel('actual value')
            plt.ylabel('predicted value')
            plt.title('LARS')
            plt.axis('equal')
        
    if saveFig: results.to_html(dirResults+'\\06-LarsResults_'+str(numFolds)+'fold_CV.html')
    mean_MSE=np.round(np.mean(results.MSE),2)
    std_MSE=np.round(np.std(results.MSE),2)
    return results, mean_MSE, std_MSE, fig1

def LARSRegressionComplete(X,y):
   
    
    model = Lars()
    mm=model.fit(X,y)
    return mm


def compareModelsCV(X,y,dirResultsMachineLearning,nFolds,models,savefig):
    modelsComparisoncols=['Model','MSE mean','MSE std']
    D_R=pd.DataFrame(columns=modelsComparisoncols)
    
        
    #Ordinary Least Square with cross validation
    if ('Ordinary least square' in models):
        results,mse_mean,mse_std, fig = ordinaryLeastSquareCV(X,y,dirResultsMachineLearning,nFolds,savefig)
        temp=pd.DataFrame([['Linear Regression with CV', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
        if savefig: fig.savefig(dirResultsMachineLearning+'\\predictionsOLS.png')
    
    
    #Ridge Regression with cross validation
    if('Ridge Regression' in models):
        results,mse_mean,mse_std, _, fig = RidgeRegressionCV(X,y,dirResultsMachineLearning,nFolds,savefig)
        temp=pd.DataFrame([['Ridge Regression with CV', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
        #if savefig: ZO_ml.ridgePath(X,y,dirResultsMachineLearning)
        if savefig: fig.savefig(dirResultsMachineLearning+'\\predictionsRidgeRegression.png')
    
    
    #Lasso Regression with cross validation
    if('Lasso Regression' in models):
        results,mse_mean,mse_std, _, fig = LassoRegressionCV(X,y,dirResultsMachineLearning,nFolds,savefig)
        temp=pd.DataFrame([['Lasso Regression with CV', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
        #if savefig: ZO_ml.lassoPath(X,y,dirResultsMachineLearning)
        if savefig: fig.savefig(dirResultsMachineLearning+'\\predictionsLAssoRegression.png')
    
    #Elastic Net Regression with cross validation
    if('Elastic Net Regression' in models):
        results,mse_mean,mse_std, _, fig = ElasticNetRegressionCV(X,y,dirResultsMachineLearning,nFolds,savefig)
        temp=pd.DataFrame([['Elastic Net Regression with CV', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if savefig: fig.savefig(dirResultsMachineLearning+'\\predictionsElasticNet.png')
    
    
    #LARS Regression with cross validation
    if('LARS Regression' in models):
        results,mse_mean,mse_std, fig = LARSregression(X,y,dirResultsMachineLearning,nFolds, savefig)
        temp=pd.DataFrame([['LARS Regression with CV', mse_mean,mse_std]],columns=modelsComparisoncols)
        D_R=D_R.append(temp)
        if savefig: fig.savefig(dirResultsMachineLearning+'\\predictionsLARS.png')
    
    plt.close('all')

    
    return D_R
'''
def compareModelsBootstrap(X,y,nBoot,dirResultsMachineLearning,nFolds,models):
    modelsComparisoncols=['Model','MSE mean','MSE std']
    D_R=pd.DataFrame(columns=modelsComparisoncols)
    
    
    #Ordinary Least Square 
    if ('Ordinary least square' in models):
        model=ordinaryLeastSquareComplete(X,y)
        D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
        temp=pd.DataFrame([['Ordinary least square', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
    
    #Ridge Regression with cross validation
    if('Ridge Regression' in models):
        _,_,_, alpha = RidgeRegressionCV(X,y,dirResultsMachineLearning,nFolds)
        model=RidgeRegressionComplete(X,y,alpha)
        D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
        temp=pd.DataFrame([['Ridge Regression', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
    
    #Lasso Regression with cross validation
    if('Lasso Regression' in models):
        _,_,_, alpha,_ = LassoRegressionCV(X,y,dirResultsMachineLearning,nFolds)
        model=LassoRegressionComplete(X,y,alpha)
        D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
        temp=pd.DataFrame([['Lasso Regression', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
    
    #ElasticNet Regression with cross validation
    if('Elastic Net Regression' in models):
        _,_,_, alpha,_ = ElasticNetRegressionCV(X,y,dirResultsMachineLearning,nFolds)
        model=ElasticNetRegressionComplete(X,y,alpha)
        D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
        temp=pd.DataFrame([['Elastic Net Regression', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 
    
    #Lars Regression 
    if('LARS Regression' in models):
        model=LARSRegressionComplete(X,y)
        D=ZO_ml.BootstrapLoop(nBoot,model,X,y)
        temp=pd.DataFrame([['LARS Regression', D.MSE[1],D.MSE[2]]],columns=modelsComparisoncols)
        D_R=D_R.append(temp) 



    return D_R
    
'''
    



