# -*- coding: utf-8 -*-
from  logproj.stat_time_series import timeStampToDays
from logproj.DIST.globalBookingAnalysis import getCoverageStats
from logproj.ml_graphs import plotGraph
import pandas as pd

def F_networkStatistics(D_mov,
                        terminalfieldFrom='LOADING_NODE',
                        terminalfieldto='DISCHARGING_NODE',
                        capacityField='QUANTITY',
                        actual=False,
                        timeColumns={}):
    #crea una cartella per i risultati dei vessel
    outputFigure={}
    sailingTime = pd.DataFrame()
    
    
    if actual == 'PROVISIONAL':
        accuracy, _ = getCoverageStats(D_mov,analysisFieldList=[timeColumns['dischargingpta'], timeColumns['loadingptd']],capacityField=capacityField)
        #calcolo le distanze (come tempi di navigazione)
        D_mov['sailingTime']=timeStampToDays(D_mov[timeColumns['dischargingpta']]-D_mov[timeColumns['loadingptd']])
        
    elif actual=='ACTUAL':
        accuracy, _ = getCoverageStats(D_mov,analysisFieldList=[timeColumns['dischargingata'], timeColumns['loadingatd']],capacityField=capacityField)
        #calcolo le distanze (come tempi di navigazione)
        D_mov['sailingTime']=timeStampToDays(D_mov[timeColumns['dischargingata']]-D_mov[timeColumns['loadingatd']])
    
    
    D_filterActual=D_mov.dropna(subset=['sailingTime'])
    sailingTime=D_filterActual.groupby([terminalfieldFrom,terminalfieldto])['sailingTime'].mean().reset_index()

    sailingTime=D_filterActual.groupby([terminalfieldFrom,terminalfieldto]).agg({'sailingTime':['mean','std','size']}).reset_index()
    sailingTime.columns = list(map(''.join, sailingTime.columns.values))
   
    fig1=plotGraph(sailingTime,terminalfieldFrom,terminalfieldto,'sailingTimemean','sailingTimesize','Network flow',arcLabel=False)
    outputFigure[f"NetworkGraph_{actual}"]=fig1
    
    sailingTime['accuracy']=[accuracy for i in range(0,len(sailingTime))]
    
    return outputFigure, sailingTime
    

       
