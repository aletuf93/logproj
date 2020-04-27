# -*- coding: utf-8 -*-
# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logproj.stat_time_series as ts
from logproj.DIST.globalBookingAnalysis import getCoverageStats
from logproj.DIST.voyageAnalysis import createTabellaMovimenti

def travelTimedistribution(D_mov,
                           capacityField='QUANTITY',
                           loadingTA='PTA_FROM',
                           loadingTD='PTD_FROM',
                           dischargingTA='PTA_TO',
                           dischargingTD='PTD_TO',
                           ):
    
    df_traveltime=pd.DataFrame(columns=['PROVISIONAL_ACTUAL','U_L_BOUND','TIME_MEAN','TIME_STD'])
    imageResults={}
    
    #get coverage
    accuracy_ub, _ = getCoverageStats(D_mov,analysisFieldList=[dischargingTD, loadingTA],capacityField=capacityField)
    
    
    #Expected travel time per container (UPPER BOUND)
    ExpectedTravelTime_ub=ts.timeStampToDays(D_mov[dischargingTD]-D_mov[loadingTA]) 
    
        
    ExpectedTravelTime_ub=ExpectedTravelTime_ub[ExpectedTravelTime_ub>0]
    mean_ExpectedTravelTime=np.mean(ExpectedTravelTime_ub)
    std_ExpectedTravelTime=np.std(ExpectedTravelTime_ub)
    
    data={'U_L_BOUND':'upperBound',
          'TIME_MEAN':mean_ExpectedTravelTime,
          'TIME_STD':std_ExpectedTravelTime,
          'accuracy':str(accuracy_ub) }
    temp=pd.DataFrame(data,index=[0])
    df_traveltime=df_traveltime.append(temp)
    
    
    
    #aspetto di graficare il LB e poi salvo
    
    
    #get coverage
    accuracy_lb, _ = getCoverageStats(D_mov,analysisFieldList=[dischargingTA, loadingTD],capacityField=capacityField)
    #Expected travel time per container (LOWER BOUND)
    ExpectedTravelTime_lb=ts.timeStampToDays(D_mov[dischargingTA]-D_mov[loadingTD])
    
    
    ExpectedTravelTime_lb=ExpectedTravelTime_lb[ExpectedTravelTime_lb>0]
    mean_ExpectedTravelTime=np.mean(ExpectedTravelTime_lb)
    std_ExpectedTravelTime=np.std(ExpectedTravelTime_lb)
    
    
    
    
    data={'U_L_BOUND':'lowerBound',
          'TIME_MEAN':mean_ExpectedTravelTime,
          'TIME_STD':std_ExpectedTravelTime,
          'accuracy':str(accuracy_lb)}
    temp=pd.DataFrame(data,index=[0])
    df_traveltime=df_traveltime.append(temp)
    
    # salvo figura
    #definisco udm
    udm='days'
    value_ub=ExpectedTravelTime_ub
    value_lb=ExpectedTravelTime_lb
    if mean_ExpectedTravelTime<1/24/60:
        udm='minutes'
        value_ub=ExpectedTravelTime_ub*24*60
        value_lb=ExpectedTravelTime_lb*24*60
    
    elif mean_ExpectedTravelTime<1: #se ho dei numeri inferiori all'unita', cambio udm
        udm='hours'
        value_ub=ExpectedTravelTime_ub*24
        value_lb=ExpectedTravelTime_lb*24
        
    fig1=plt.figure()
    plt.hist(value_ub,color='orange')
    plt.hist(value_lb,color='blue',alpha=0.6)
    plt.title(f"Travel time ({udm})")
    plt.xlabel(f"{udm}")
    plt.ylabel('Quantity')
    plt.legend(['Upper bound','Lower bound'])
    
    imageResults[f"travel_time_per_movement"]=fig1
        
    return imageResults, df_traveltime

# %%
def itemLifeCycle(D_mov,itemfield='CONTAINER', 
                  locationfrom='LOADING_NODE',
                  locationto='DISCHARGING_NODE',
                  capacityField='QUANTITY',
                  timeColumns={},
                  sortTimefield='PTA_FROM',
                  numItemTosave=1):
    # costruisce il ciclo di vita di carico/scarico per ogni itemfield. Gli itemfield devono rappresentare prodotti/unita' di
    #carico fisicamente diverse
    
    df_lifeCycle={}
    figureOutput={}
    
    #verifico di avere tutte le colonne necessarie
    if all( column in timeColumns.keys() for column in ['loadingpta','loadingptd','dischargingpta','dischargingptd']):
        
        
        #Container lifeCycle
        D_movLifeCycle=D_mov.groupby([itemfield]).size().reset_index()
        D_movLifeCycle=D_movLifeCycle.rename(columns={0:'Movements'})
        D_movLifeCycle=D_movLifeCycle.sort_values(['Movements'],ascending=False).reset_index()
        for j in range(0,min(numItemTosave,len(D_movLifeCycle))):
        
            itemName = D_movLifeCycle[itemfield].iloc[j]
            mostTravelled=D_movLifeCycle[itemfield][j]
            MostTravelledMovements=D_mov[D_mov[itemfield]==mostTravelled]
            MostTravelledMovements=MostTravelledMovements.sort_values([sortTimefield]).reset_index()
            
            
            #identifico la copertura
            allcolumns = [itemfield,timeColumns['loadingpta'],timeColumns['loadingptd'],timeColumns['dischargingpta'],timeColumns['dischargingptd']]
            accuracy, _ = getCoverageStats(MostTravelledMovements,analysisFieldList=allcolumns,capacityField=capacityField)
            MostTravelledMovements['accuracy'] = [accuracy for i in range(0,len(MostTravelledMovements))]
            df_lifeCycle[f"lifeCycle_{itemName}"]=MostTravelledMovements
        
        
        
            #Trasformarlo in movimenti singoli (come per le analisi di capacità)
            D_movimentiPerContainer = createTabellaMovimenti( MostTravelledMovements,
                                locfrom = locationfrom,                     
                                locto= locationto,
                                capacityField=capacityField,
                                timeColumns=timeColumns
                                )
            
            
            D_movimentiPerContainer=D_movimentiPerContainer.sort_values(['PTA'])
            #D_movimentiPerContainer=D_movimentiPerContainer[~(D_movimentiPerContainer.Type=='Transit')]
            
            cols=['DateTime','Location','value'];
            graficoLifeCycle=pd.DataFrame(columns=cols)
            
            for i in range(0,len(D_movimentiPerContainer)):
                movimento=D_movimentiPerContainer.iloc[i,:]
                if(movimento.InOut=='IN'):
                    temp=pd.DataFrame([[movimento.PTA,movimento.Location, 0.5]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
                    temp=pd.DataFrame([[movimento.PTD,movimento.Location, 0.5]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
                    temp=pd.DataFrame([[movimento.PTD+pd.to_timedelta(1, unit='s'),movimento.Location, 1]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
                elif(movimento.InOut=='OUT'):
                    temp=pd.DataFrame([[movimento.PTA,movimento.Location, 0.5]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
                    temp=pd.DataFrame([[movimento.PTD,movimento.Location, 0.5]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
                    temp=pd.DataFrame([[movimento.PTD+pd.to_timedelta(1, unit='s'),movimento.Location, 0]],columns=cols);
                    graficoLifeCycle=graficoLifeCycle.append(temp)
            
            fig1=plt.figure(figsize=(20,10))
            plt.step(graficoLifeCycle.DateTime,graficoLifeCycle.value,where='post',color='orange')
            plt.xticks(rotation=30)
            plt.xlabel('timeline')
            plt.ylabel('status')
            plt.title('Itemfield: '+str(mostTravelled)+' life cycle')    
            #fig1.savefig(dirResults+'\\02-ContainerLifeCycle'+str(mostTravelled)+'.png')
            figureOutput[f"loadingUnloading_itemfield_{itemName}"]=fig1
        
            graficoLifeCycle['distance']=0
            #creo grafico spazio-tempo
            for i in range(1,len(graficoLifeCycle)):
                movPrecedente=graficoLifeCycle.iloc[i-1]
                movCurrent=graficoLifeCycle.iloc[i]
                if movCurrent.Location==movPrecedente.Location:
                    graficoLifeCycle.iloc[i,graficoLifeCycle.columns.get_loc('distance')]= graficoLifeCycle.distance.iloc[i-1]
                else:
                    graficoLifeCycle.iloc[i,graficoLifeCycle.columns.get_loc('distance')]= graficoLifeCycle.distance.iloc[i-1]+1
            
            fig1=plt.figure(figsize=(20,10))
            plt.plot(graficoLifeCycle.distance,graficoLifeCycle.DateTime,color='orange')
            plt.xlabel('distance')
            plt.ylabel('timeline')
            plt.title('Container: '+str(mostTravelled)+' time-distance graph') 
            figureOutput[f"spaceTime_itemfield_{itemName}"]=fig1
    else: 
        print(f"WARNING: NO PTA AND PTD FOR ITEM {itemName}")
    return figureOutput, df_lifeCycle

# %% level of service

def calculateLoS(D_mov,
                           capacityField='QUANTITY',
                           timeColumns={}
                           ):
    
    output_figure={}
    coverages=pd.DataFrame()
    
    
    if all( column in timeColumns.keys() for column in ['loadingptd','dischargingpta',
                                                        'loadingatd','dischargingata']):
        columnsNeeded = [timeColumns['loadingptd'], timeColumns['dischargingpta'],
                         timeColumns['loadingatd'], timeColumns['dischargingata']]
        
        accuracy, _ = getCoverageStats(D_mov,analysisFieldList=columnsNeeded,capacityField=capacityField)
        
        D_time = D_mov.dropna(subset=columnsNeeded)
        
        plannedTime =  D_time[timeColumns['dischargingpta']] - D_time[timeColumns['loadingptd']]
        actualTime =   D_time[timeColumns['dischargingata']] -  D_time[timeColumns['loadingatd']]
        
        Los = actualTime<plannedTime
        D_res = Los.value_counts()
        
        fig1=plt.figure()
        plt.pie(D_res,autopct='%1.1f%%', shadow=True, startangle=90,labels=D_res.index)
        plt.title('Level of Service')
        
        output_figure['level_of_service']=fig1
        
        coverages=pd.DataFrame([accuracy])
        
    return output_figure, coverages
        
