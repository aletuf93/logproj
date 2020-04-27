# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def getAdvanceInPlanning(D_mov,loadingptafield='LOADING_TIME_WINDOWS_PROVISIONAL_START'):
    #la funzione calcola la distribuzione del tempo di anticipo di pianificazione
    #come differenza fra il timestamp_ in e la finestra temporale di carico
    
    output_figure={}
    output_data={}
    output_coverage={}
    
    #filterNan
    D_mov_filtered = D_mov[['TIMESTAMP_IN',loadingptafield]].dropna()
    
    if len(D_mov_filtered)==0:
        return output_figure, pd.DataFrame(['No PTA fields to perform this analysis']) 
    if loadingptafield=='TIMESTAMP_IN': #se uso la stessa colonna ottengo 0
        mean_advanceInPlanning =  std_advanceInPlanning =0
        advanceInPlanningDistribution =[]
    else:
        advanceInPlanning=D_mov_filtered[loadingptafield]-D_mov_filtered['TIMESTAMP_IN']
        advanceInPlanningD=advanceInPlanning.dt.components['days']
        advanceInPlanningH=advanceInPlanning.dt.components['hours']
        advanceInPlanningM=advanceInPlanning.dt.components['minutes']
        advanceInPlanning=advanceInPlanningD + advanceInPlanningH/24 +advanceInPlanningM/(60*24)
        advanceInPlanning=advanceInPlanning[advanceInPlanning>0]
        mean_advanceInPlanning=np.mean(advanceInPlanning)
        std_advanceInPlanning=np.std(advanceInPlanning)
        advanceInPlanningDistribution=advanceInPlanning
    
    
    
    if len(advanceInPlanningDistribution)>0:
        #Advance in planning        
        fig_planningAdvance=plt.figure()
        plt.title('Days of advance in booking')
        plt.hist(advanceInPlanning,color='orange',bins=np.arange(0,max(advanceInPlanningDistribution),1))
        plt.xlabel('days')
        plt.ylabel('N.ofBooks')
        
        #output_figure
        output_figure['ADVANCE_IN_PLANNING']=fig_planningAdvance
    
    #output_data
    output_data['ADVANCE_PLANNING_MEAN']= mean_advanceInPlanning
    output_data['ADVANCE_PLANNING_STD']=std_advanceInPlanning
    output_data['SERIES']=advanceInPlanningDistribution
    
    #get coverage
    output_coverage['ADVANCE_PLANNING_MEAN'] = getCoverageStats(D_mov,loadingptafield,capacityField='QUANTITY')
    output_coverage['ADVANCE_PLANNING_STD']=output_coverage['ADVANCE_PLANNING_MEAN']
    output_coverage['SERIES']=output_coverage['ADVANCE_PLANNING_MEAN']   
    
    D_global=pd.DataFrame([output_data,output_coverage]).transpose()
    D_global.columns=['VALUE','ACCURACY']
    
    
    return output_figure, D_global 

# %%
def getCoverageStats(D_mov,analysisFieldList,capacityField='QUANTITY'):
    #ritorna le statistiche di copertura e il numero di elementi
    #le statistiche di copertura sono una tupla di due elementi,
    #indicanti la copertura percentuale sul numero di linee e le quantita' rispettivamente
    n_lines=len(D_mov)
    tot_qties = np.nansum(D_mov[capacityField])   
    
    
    tot_lines=len(D_mov[analysisFieldList].dropna())
    
    #se vedo verificare una copertura su una lista di attributi
    if isinstance(analysisFieldList,list):
        listCol=analysisFieldList
        listCol.append(capacityField)
        D_filtered_qties=D_mov[listCol].dropna()
    
#se devo verificare la copertura su un singolo attributo    
    else:
        D_filtered_qties=D_mov[[analysisFieldList,capacityField]].dropna()
    
    lineCoverage = tot_lines/n_lines
    
    #la copertura sulle quantita' funziona solo se analysisFieldList e capacityField
    # sono diversi, altrimenti somma due volte la stessa colonna
    if capacityField==analysisFieldList:
        qtyCoverage=1
    else:
        qtyCoverage =  np.nansum(D_filtered_qties[capacityField])/tot_qties
    
    return (lineCoverage,qtyCoverage), tot_lines
    
    


# %%
def movementStatistics(D_mov, capacityField='QUANTITY'):
    #this function performs global analysis on the D_mov dataframe
    #returning a dataframe with global statistics
    
    data={}
    coverage_stats={}
      
    
    for col in D_mov.columns:
        # per tutti calcolo le statistiche di conteggio
        coverage_stats[f"COUNT.{col}"], nrows = getCoverageStats(D_mov,col,capacityField)
        if any( [ isinstance(i ,dict) for i in D_mov[col]]) : 
            data[f"COUNT.{col}"] = nrows
        else:
            data[f"COUNT.{col}"] = len(D_mov[col].unique())
        # se e' un numero calcolo le statistiche di somma
        if (D_mov[col].dtypes==np.float) |(D_mov[col].dtypes==np.int):
            data[f"SUM.{col}"] = np.nansum(D_mov[col])
            coverage_stats[f"SUM.{col}"] = coverage_stats[f"COUNT.{col}"]
        
        # se e' una data identifico il numero di giorni, il primo e l'ultimo giorno
        if (D_mov[col].dtypes==np.datetime64) |  (D_mov[col].dtypes=='<M8[ns]'):
            BookingDates = np.unique(D_mov[col].dt.date)
            beginningTimeHorizon=min(BookingDates)
            endTimeHorizon=max(BookingDates)
            NofBookingDays=len(BookingDates)
            
            data[f"N.OF.DAYS.{col}"]=NofBookingDays
            data[f"FIRST.DAY.{col}"]= beginningTimeHorizon
            data[f"LAST.DAY.{col}"]= endTimeHorizon
            
            coverage_stats[f"N.OF.DAYS.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"FIRST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"LAST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]
    
    
    D_global=pd.DataFrame([data,coverage_stats]).transpose()
    D_global.columns=['VALUE','ACCURACY']
    return  D_global 