#This package prepare data and develops statistic (diagnostic) analysis on the warehouse
import pandas as pd
import logproj.ml_machine_learning as ml #import machine learning package

def cleanDatatableLearningOrders(D_tem,features):
    #import a dataTable with attributes:
    #features is a list of feature that are used to clean the data. For each feature
    #an IQR is defined and the outlier are removed
    #and clean data using the interquartile range method
    
    #set default error variable
    D_res_tem_IN=pd.DataFrame()
    D_res_tem_OUT=pd.DataFrame()
    perc_IN=[]
    perc_OUT=[]
    
    D_tem_IN=D_tem[D_tem['INOUT']=='+']
    D_tem_OUT=D_tem[D_tem['INOUT']=='-']


    ################ DATA CLEANING #############

    if len(D_tem_IN)>0:
        
        #INBOUND
        for feat in features: # remove zero values
            D_tem_IN=D_tem_IN[(D_tem_IN[feat] != 0)]
        #D_tem_IN=D_tem_IN.dropna().reset_index()

        #clean using IQR
        if len(D_tem_IN)>0:
            D_res_tem_IN, perc_IN=ml.cleanUsingIQR(D_tem_IN,features)

    if len(D_tem_OUT)>0:
        #OUTBOUND
        for feat in features:
            D_tem_OUT =D_tem_OUT[(D_tem_OUT[feat] != 0)]
        #D_tem_OUT=D_tem_OUT.dropna().reset_index()

        #clean using IQR
        if len(D_tem_OUT)>0:
            D_res_tem_OUT, perc_OUT=ml.cleanUsingIQR(D_tem_OUT,features)
    return D_res_tem_IN, D_res_tem_OUT, perc_IN, perc_OUT
