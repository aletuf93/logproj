# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from logproj.P8_performanceAssessment.vehicle_assessment import createTabellaMovimenti
from logproj.P8_performanceAssessment.utilities_movements import getCoverageStats

# %%
def itemSharePieGraph(D_mov, itemfield,capacityField ='QUANTITY'):

    #itemfield rappresenta i codici di famiglie di prodotto per rappresentarne la quota di mercato
    #capacityField e' un campo di capacita' per calcolare le coperture

    #calcolo le coperture
    accuracy, _ = getCoverageStats(D_mov,itemfield,capacityField='QUANTITY')

    #TEU-FEU share
    D_movType=D_mov.groupby([itemfield]).size().reset_index()
    D_movType=D_movType.rename(columns={0:'Percentage'})
    labels=D_movType[itemfield]
    sizes=D_movType.Percentage
    explode = 0.1*np.ones(len(sizes))
    fig1, ax1 = plt.subplots(figsize=(20,10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    #creo tabella per tipo di container
    D_movCode=D_mov.groupby([itemfield]).size().reset_index()
    D_movCode=D_movCode.rename(columns={0:'Quantity'})
    D_movCode=D_movCode.sort_values(['Quantity'],ascending=False)
    #D_movCode.to_excel(dirResults+'\\02-ContainerTypeStats.xlsx')

    D_movCode['accuracy']=[accuracy for i in range(0,len(D_movCode))]

    return fig1, D_movCode


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
        print(f"WARNING: NO PTA AND PTD")
    return figureOutput, df_lifeCycle
