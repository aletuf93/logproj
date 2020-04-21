# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


#machine learning packages
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold,SelectFromModel


from logproj.ml_dataCleaning import dummyColumns
from logproj.ml_expore import correlationMatrix
from logproj.ml_regression_linear_models import LassoCV, fit_linear_reg

def diagnoseForFeatureSelection(X,y,dirResultsMachineLearning,diagnoseModels):
    Q=dummyColumns(X)
    Y=pd.concat([Q,y], axis=1)
    plt.close('all')

    #Analizzo la correlazione fra le variabili
    correlationMatrix(Y,dirResultsMachineLearning,annotationCell=False)

    #analysis of correlation with the target variable
    if  diagnoseModels.count('correlation')>0:
        selectByCorrelation(X,y,1,dirResultsMachineLearning,True)

    #analysis of the variance of the variables
    if diagnoseModels.count('variance')>0:
        selectByVariance(X, 1,dirResultsMachineLearning,True)

    #analysi of the selection of the variables by lasso
    if diagnoseModels.count('lasso')>0:
        selectByLassoL1(X,y,1,dirResultsMachineLearning,True)

    #Verifico se posso esprimere la varianza in un sottoInsieme di componenti principali
    if diagnoseModels.count('PCA')>0:
        PCAplot(len(Q.columns),Q,dirResultsMachineLearning, diagnose=True)

    #Creo la curva del forward Stepwise
    if diagnoseModels.count('forward stepwise')>0:
        selectByForwardStepwiseSelection(X,y,dirResultsMachineLearning,len(X.columns),saveFig=True )

    return True

def selectPreprocessFeature(X,y,value,model):
    #X is the feature dataframe
    #y is the target variable
    #value indicates a parameter of the model (if any)
        #for correlation it is the minimum treshold
        #for forward stepwise it is the number of features to select
        # for PCA it is the number of principal component
    if model=='correlation':
        return selectByCorrelation(X,y,value,'',False)
    elif model=='variance':
        return selectByVariance(X, value,'',False)
    elif model=='lasso':
        return selectByLassoL1(X,y,value,'',False)
    elif model=='tree':
        return selectByTree(X,y)
    elif model=='forward stepwise':
        return selectByForwardStepwiseSelection(X,y,'', value,False)
    elif model=='PCA':
        return PCAplot(value,X,'', False)
    else: #if no feature selection, the dataset is only returned as numerical (dummy) to code strings
        return dummyColumns(X)


def selectByCorrelation(X,y,corrThreshold, dirResults,diagnose):
    #select only the features (column) of a dataframe having a correlation
    #>= than the corrThreshold with the target variable
    targetVariable=y.name
    Q=dummyColumns(X)
    Q=pd.concat([Q,y], axis=1)
    cor = Q.corr()
    cor_target = abs(cor[targetVariable])


    if diagnose:
        numFeatSelected=[]
        for i in range(1,101):
            val=i/100
            nn=len(cor_target[cor_target>val])
            #print(nn)
            numFeatSelected.append(nn)
        plt.plot(range(1,101), numFeatSelected)
        plt.title('Correlation graph')
        plt.xlabel('Corr threshold %')
        plt.ylabel('Num. selected features')
        plt.savefig(dirResults+'\\CorrelationChart.png')
        plt.close('all')

    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>corrThreshold]
    relevant_features=list(relevant_features.index.values)
    try: #check if the target feature is contained within the relevant features and remove
        relevant_features.remove(targetVariable)
    except:
        True
    Z=Q.loc[:,relevant_features]
    return Z

def selectByVariance(X, perc,dirResults,diagnose):

    #It select the features having the same value in more than perc of the samples
    Q=dummyColumns(X)
    if diagnose:
        numFeatSelected=[]
        for i in range(1,101):
            val=i/100
            sel=VarianceThreshold(threshold=(val))
            sel.fit_transform(Q)
            nn=len(Q.columns[sel.get_support()])
            #print(nn)
            numFeatSelected.append(nn)
        plt.plot(range(1,101), numFeatSelected)
        plt.title('Variance graph')
        plt.xlabel('Variance threshold %')
        plt.ylabel('Num. selected features')
        plt.savefig(dirResults+'\\VarianceChart.png')
        plt.close('all')

    sel=VarianceThreshold(threshold=(perc))
    Z=sel.fit_transform(Q)
    feature_idx = sel.get_support()
    feature_name = Q.columns[feature_idx]
    res=pd.DataFrame(Z,columns=feature_name)
    return res

def selectByLassoL1(X,y,value,dirResults,diagnose):
    #lasso feature selection works for regression models
    Q=dummyColumns(X)
    clf = LassoCV(cv=5)

    if diagnose:
        numFeatSelected=[]
        for i in range(1,101):
            val=i/100
            sfm = SelectFromModel(clf, threshold=val)
            sfm.fit(Q, y)
            nn=len(Q.columns[sfm.get_support()])
            #print(nn)
            numFeatSelected.append(nn)
        plt.plot(range(1,101), numFeatSelected)
        plt.title('Lasso graph')
        plt.xlabel('Coeff threshold %')
        plt.ylabel('Num. selected features')
        plt.savefig(dirResults+'\\LassoChart.png')
        plt.close('all')

    #_, _, _, alphaValue, _= LassoRegressionCV(Q,y,'',nFolds=5, saveFig=False) #previous version with lasso from ZO_RegressionLinearModel
    #las=LassoRegressionComplete(Q,y,alphaValue)

    sfm = SelectFromModel(clf, threshold=value)
    sfm.fit(Q, y)
    #n_features = sfm.transform(X).shape[1]
    #model = SelectFromModel(las, prefit=True,threshold=0.25)

    Z = sfm.transform(Q)
    feature_idx = sfm.get_support()
    feature_name = Q.columns[feature_idx]
    res=pd.DataFrame(Z,columns=feature_name)
    return res

def selectByTree(X,y):
    #lasso feature selection works for regression models
    Q=dummyColumns(X)
    tree = ExtraTreesClassifier(n_estimators=50).fit(Q,y)
    model = SelectFromModel(tree, prefit=True)
    Z = model.transform(Q)
    feature_idx = model.get_support()
    feature_name = Q.columns[feature_idx]
    res=pd.DataFrame(Z,columns=feature_name)
    return res

def ridgePath(XX,y,dirResults):

    X=dummyColumns(XX)
    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=[]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)


    # Display results
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
    plt.legend(columnNames)
    plt.savefig(dirResults+'\\03_RidgePath.png')
    plt.close('all')
    return True

def lassoPath(XX,y,dirResults):

    X=dummyColumns(XX)

    columnNames=list(pd.concat([X,y], axis=1))
    columnNames[-1]=[]

    alphas = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]

    coefs = []
    for a in alphas:
        ridge = linear_model.Lasso(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)


    # Display results
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
    plt.legend(columnNames)
    plt.savefig(dirResults+'\\04_LassoPath.png')
    plt.close('all')
    return True


def selectByForwardStepwiseSelection(XX,y,dirResults,n_feat,saveFig=False):


    #Converto eventuali variabili categoriche
    X=dummyColumns(XX)
    k = len(X.columns)

    columnNames=['Features','RSS','R_squared']
    resultFSs=pd.DataFrame(columns=columnNames)

    remaining_features = list(X.columns.values)
    features = []
    numb_features = [np.inf]
    RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
    features_list = dict()

    for i in range(1,k+1):
        best_RSS = np.inf

        for combo in itertools.combinations(remaining_features,1):


                RSS = fit_linear_reg(X[list(combo) + features],y)   #Store temp result

                if RSS[0] < best_RSS:
                    best_RSS = RSS[0]
                    best_R_squared = RSS[1]
                    best_feature = combo[0]

        #Updating variables for next loop
        features.append(best_feature)
        if len(remaining_features)>0:
            remaining_features.remove(best_feature)

        #Saving values for plotting
        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        features_list[i] = features.copy()

        listResult=[]
        listResult.append(features_list[i])
        listResult.append([round(best_RSS,3)])
        listResult.append([round(best_R_squared,3)])
        listResult=[listResult]
        row=pd.DataFrame(listResult,columns=columnNames)
        resultFSs=resultFSs.append(row)

        numb_features.append(len(features))
    #resultFSs.to_excel(dirResults+'\\00-ForwardStepwiseSelection.xlsx')

    #Salvo le variabili
    s =resultFSs['Features']
    mlb = preprocessing.MultiLabelBinarizer()
    aaa=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=resultFSs.index)
    bbb=pd.concat([resultFSs,aaa],axis=1)
    if saveFig:
        bbb.to_excel(dirResults+'\\00-ForwardStepwiseSelection.xlsx')

    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list})
    df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
    #df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
    df_min.to_excel(dirResults+'\\00-00_ForwardStepwiseSelection_min.xlsx')


    df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
    df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)

    if saveFig:
        fig = plt.figure(figsize = (16,6))
        ax = fig.add_subplot(1, 2, 1)

        ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
        ax.set_xlabel('# Features')
        ax.set_ylabel('RSS')
        ax.set_title('RSS - Forward Stepwise selection')
        ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
        ax.legend()

        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
        ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
        ax.set_xlabel('# Features')
        ax.set_ylabel('R squared')
        ax.set_title('R_squared - Forward Stepwise selection')
        ax.legend()
        plt.show()
        fig.savefig(dirResults+'\\00_ForwardStepwiseSelection.png')
        plt.close('all')
    result=resultFSs.reset_index()
    selected_features=result.Features[n_feat-1]

    X_res=X.loc[:,selected_features]
    return X_res

''' too slow
def bestSubsetSelection(X,y,dirResults):

    #Converto eventuali variabili categoriche
    for column in X.columns:
     if X[column].dtype==object:
         dummyCols=pd.get_dummies(X[column])
         X=pd.concat([X,dummyCols], axis=1)
         del X[column]

    k = len(X.columns)

    RSS_list, R_squared_list, feature_list = [],[], []
    numb_features = []

    #Looping over k = 1 to k = 11 features in X
    for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):

        #Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns,k):
            tmp_result = fit_linear_reg(X[list(combo)],y)   #Store temp result
            RSS_list.append(tmp_result[0])                  #Append lists
            R_squared_list.append(tmp_result[1])
            feature_list.append(combo)
            numb_features.append(len(combo))



    #Salvo le variabili da testare con un dataset piccolo
    #s =resultFSs['Features']
    #mlb = MultiLabelBinarizer()
    #aaa=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=resultFSs.index)
    #bbb=pd.concat([resultFSs,aaa],axis=1)
    #bbb.to_html(dirResults+'\\00-ForwardStepwiseSelection2.html')

    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
    df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
    #df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
    df_min.to_html(dirResults+'\\00-BestSubSetSelection_min.html')


    df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
    df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)

    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
    ax.set_xlabel('# Features')
    ax.set_ylabel('RSS')
    ax.set_title('RSS - Best subset selection')
    ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
    ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
    ax.set_xlabel('# Features')
    ax.set_ylabel('R squared')
    ax.set_title('R_squared - Best subset selection')
    ax.legend()
    plt.show()
    fig.savefig(dirResults+'\\00_BestSubsetSelection.pdf')
    return True
'''
# In[1]: DIMENSIONALITY REDUCTION FUNCTIONS
def PCAplot(n_comp,XX,dirResults, diagnose=True):
    #n_comp is the number of component of the PCA
    #XX is the dataframe to build the PCA on
    #dirResult is a directory path where to save the plot (only if diagnose==true)
    #diagnose perform an analysis on the percentage of variance explained increasing the number of component
    #if the number of component is 2 and diagnose is true the PCA 2-dim graph is saved
    D_Table=dummyColumns(XX)
    # applico la PCA
    data_scaled = pd.DataFrame(preprocessing.scale(D_Table),columns = D_Table.columns)
    pca = PCA(n_components=n_comp)
    PC=pca.fit_transform(data_scaled)

    # Salvo i coefficienti dei parametri
    #components= pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])

    if diagnose:
        components= pd.DataFrame(pca.components_,columns=data_scaled.columns)
        components.to_excel(dirResults+'\\PCA.xlsx')
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

        #Plot variance explaination
        fig=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(30,100.5)
        plt.style.context('seaborn-whitegrid')
        plt.plot(var)
        fig.savefig(dirResults+'\\PCA_varianceExplanation.png')

        # Plot graph PCA
        if n_comp==2: #se ho solo due comèpnenti posso graficare
            fig1=plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.scatter(PC[:, 0], PC[:, 1],color='orange')
            plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
            plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
            plt.axis('equal')
            plt.tight_layout()
            plt.title('Principal Component plot')
            fig1.savefig(dirResults+'\\PCA_plot.png')
            plt.close('all')
    return pd.DataFrame(PC)

