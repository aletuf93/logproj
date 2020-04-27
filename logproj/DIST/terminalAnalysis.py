# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from logproj.DIST.voyageAnalysis import createTabellaMovimenti, defineRouteTable
from logproj.DIST.globalBookingAnalysis import getCoverageStats, getAdvanceInPlanning

# %%
def createTabellaProductivityAllocationTerminal(D_mov,
                                                timefield='TIMESTAMP_IN',
                                                locfrom = 'LOADING_NODE',                     
                                                locto= 'DISCHARGING_NODE',
                                                capacityField='QUANTITY',
                                                voyagefield='VEHICLE_CODE',
                                                timeColumns={},
                                                censoredData=False,
                                                actual='PROVISIONAL',
                                                splitInOut=True):
    
   
    D=createTabellaMovimenti(D_mov=D_mov,
                            locfrom = locfrom,
                            locto= locto,
                            capacityField=capacityField,
                            timeColumns=timeColumns)
    
    
    #ricostruisco le rotte
    D_route, timestartfield, timeendfield = defineRouteTable(D,agregationVariables =[voyagefield,'InOut'],actual=actual)
    
    
    if splitInOut: #separate inbound and outbound productivity
       D_terminal, timestartfield, timeendfield = defineRouteTable(D,agregationVariables =[voyagefield,'InOut'],actual=actual)
    else:           # merge inbound and outbound productivity
       D_terminal, timestartfield, timeendfield = defineRouteTable(D,agregationVariables =[voyagefield,],actual=actual)
    
    if len(D_terminal)==0:
        print("No terminal data")
        return []
    
    #identifico il primo giorno di planning
    firstPlanningDay=min(D_mov[timefield].dt.date)  
    
    #Identifico l'intervallo medio di anticipo in giorni sulle prenotazioni
    _, df_advance  = getAdvanceInPlanning(D_mov,loadingptafield=timeColumns['loadingpta'])
    mean_advanceInPlanning=df_advance.loc['ADVANCE_PLANNING_MEAN']['VALUE']
    std_advanceInPlanning=df_advance.loc['ADVANCE_PLANNING_STD']['VALUE']
    lowerBoundDataCensored=firstPlanningDay+pd.Timedelta(days=(mean_advanceInPlanning+std_advanceInPlanning))
    
    
    #Identifico l'ultimo giorno di planning
    lastPlanningDay=max(D_mov[timefield].dt.date)
    
    #rimuovo i movimenti al di fuori dell'orizzonte di riferimento
    if(not(censoredData)): #se non voglio tenere conto dei dati censurati
            D_terminal=D_terminal[(D_terminal[timestartfield]>pd.to_datetime(lowerBoundDataCensored)) & (D_terminal[timeendfield]<pd.to_datetime(lastPlanningDay))]
            D_terminal = D_terminal.reset_index(drop=True)
    
    allocationDriverPerHour=D_terminal[timeendfield]-D_terminal[timestartfield]
    allocationDriverPerHourD=allocationDriverPerHour.dt.components['days']
    allocationDriverPerHourH=allocationDriverPerHour.dt.components['hours']
    allocationDriverPerHourM=allocationDriverPerHour.dt.components['minutes']
    allocationDriverPerHour=allocationDriverPerHourD*24 + allocationDriverPerHourH +allocationDriverPerHourM/(60)
    
    D_terminal['HoursAllocation']=allocationDriverPerHour
    D_terminal['CurrentCapacity']=np.abs(D_terminal['Movementquantity'])/(D_terminal['HoursAllocation'])
    
    #assegno il giorno della settimana
    #DayOfTheWeek,weekend=ts.assegnaGiornoSettimana(D_terminal,'PTA')
    #D_terminal['dayOfTheWeek']=DayOfTheWeek
    #D_terminal['weekend']=weekend
    
    #assegno la produttività oraria
    handlingTime=D_terminal[timeendfield]-D_terminal[timestartfield]     
    handlingTimeH=handlingTime.dt.components['hours']
    handlingTimeM=handlingTime.dt.components['minutes']
    handlingTime=handlingTimeH +handlingTimeM/(60)
    D_terminal['handlingTime']=handlingTime
    
    
    hourProd=D_terminal['Movementquantity']/handlingTime
    D_terminal['hourProductivity']=hourProd
    
    
    return D_terminal

# %%
def E_terminalStatistics(D_mov,
                        timefield='TIMESTAMP_IN',
                        locfrom = 'LOADING_NODE',                     
                        locto= 'DISCHARGING_NODE',
                        voyagefield='VEHICLE_CODE',
                        capacityField='QUANTITY',
                        timeColumns={},
                        censoredData=False,
                        actual='PROVISIONAL',
                        splitInOut=True):
    
    
    outputfigure={}
    D_terminal=pd.DataFrame()
    
    #calcolo coperture e verifico colonne in input
    if actual=='PROVISIONAL':
        colonneNecessarie = ['loadingpta','loadingptd','dischargingpta','dischargingptd']
        if all([column in timeColumns.keys()  for column in colonneNecessarie ]):
            allcolumns = [locfrom,locto, timeColumns['loadingpta'],timeColumns['loadingptd'],timeColumns['dischargingpta'],timeColumns['dischargingptd']]
            accuracy, _ = getCoverageStats(D_mov,analysisFieldList=allcolumns,capacityField='QUANTITY')
        else:
            colonneMancanti=[column   for column in  colonneNecessarie if column not in timeColumns.keys()]
            D_coverages=pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
    elif actual == 'ACTUAL':
        colonneNecessarie = ['loadingata','loadingatd','dischargingata','dischargingatd']
        if all([column in timeColumns.keys()  for column in colonneNecessarie ]):
            allcolumns = [locfrom,locto, timeColumns['loadingata'],timeColumns['loadingatd'],timeColumns['dischargingata'],timeColumns['dischargingatd']]
            accuracy, _ = getCoverageStats(D_mov,analysisFieldList=allcolumns,capacityField='QUANTITY')
        else:
            colonneMancanti=[column   for column in  colonneNecessarie if column not in timeColumns.keys()]
            D_coverages=pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
    #assegno accuratezza
    D_coverages = pd.DataFrame(accuracy)
    
    #creo tabella allocation productivity terminal
    D_terminal=createTabellaProductivityAllocationTerminal (D_mov,
                                                            timefield=timefield,
                                                            locfrom = locfrom,                     
                                                            locto= locto,
                                                            capacityField=capacityField,
                                                            voyagefield=voyagefield,
                                                            timeColumns=timeColumns,
                                                            censoredData=censoredData,
                                                            actual=actual,
                                                            splitInOut=splitInOut)
    
    
        
        
    
    BookingTrendcols=['Terminal','00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    D_bookingTerminal=pd.DataFrame(columns=BookingTrendcols)
    Terminals=np.unique(D_terminal.Location)
    for i in range(0,len(Terminals)):
        
        #Cerco correlazione fra tempo di handling e quantità
        terminal=Terminals[i]
        dataTerminalTemp=D_terminal[D_terminal.Location==terminal]
                
        for hh in ['IN','OUT']:
            dataTerminal=dataTerminalTemp[dataTerminalTemp.InOut==hh]
            dataTerminal=dataTerminal.dropna()
            dataTerminal=dataTerminal[dataTerminal['handlingTime']>0] #rimuovo i tempi di movimentazione nulli
            if(len(dataTerminal)>1):
                
                #Analisi di regressione semplice
                fig1=plt.figure()
                sns.regplot(dataTerminal['Movementquantity'], dataTerminal['handlingTime'], color='orange', marker="o")
                plt.ylabel('Handling Time')
                plt.xlabel('Handled quantity '+hh)
                plt.title(hh+' Terminal: '+str(terminal))
                outputfigure[f"productivity_IN_regression_{terminal}"]=fig1
                
                
                              
                
                #traccio l'analisi in frequenza di quei rapporti (produttività oraria)
                
                fig2=plt.figure()
                #bins=np.arange(0,max(np.sqrt(max(np.abs(dataTerminal['hourProductivity']))),1))
                plt.hist(dataTerminal['hourProductivity'],color='orange')
                plt.ylabel('Frequency')
                plt.xlabel(hh+' Movements per hour')
                plt.title('Productivity '+hh+' Terminal : '+str(terminal))
                outputfigure[f"productivity_IN_pdf_{terminal}"]=fig2
                plt.close('all')  
                
                
                #Identifico il trend sulle time windows
                for j in range(0,len(dataTerminal)):
                    
                    #azzero le statistiche orarie
                    H_00=0
                    H_01=0
                    H_02=0
                    H_03=0
                    H_04=0
                    H_05=0
                    H_06=0
                    H_07=0
                    H_08=0
                    H_09=0
                    H_10=0
                    H_11=0
                    H_12=0
                    H_13=0
                    H_14=0
                    H_15=0
                    H_16=0
                    H_17=0
                    H_18=0
                    H_19=0
                    H_20=0
                    H_21=0
                    H_22=0
                    H_23=0
                    
                    caricoScarico=dataTerminal.iloc[j]
                    if actual=='PROVISIONAL':
                        istInizio=caricoScarico.PTA
                        istFine=caricoScarico.PTD
                    elif actual=='ACTUAL':
                        istInizio=caricoScarico.ATA
                        istFine=caricoScarico.ATD
                    qty=caricoScarico.CurrentCapacity
                    oraInizio=istInizio.hour
                    oraFine=istFine.hour
                    
                     
                    
                    if(oraFine>oraInizio):
                        for k in range(oraInizio,oraFine+1):
                            if k==0:
                                H_00=H_00+qty
                            elif k==1:
                                H_01=H_01+qty
                            elif k==2:
                                H_02=H_02+qty
                            elif k==3:
                                H_03=H_03+qty
                            elif k==4:
                                H_04=H_04+qty
                            elif k==5:
                                H_05=H_05+qty
                            elif k==6:
                                H_06=H_06+qty
                            elif k==7:
                                H_07=H_07+qty
                            elif k==8:
                                H_08=H_08+qty
                            elif k==9:
                                H_09=H_09+qty
                            elif k==10:
                                H_10=H_10+qty
                            elif k==11:
                                H_11=H_11+qty
                            elif k==12:
                                H_12=H_12+qty
                            elif k==13:
                                H_13=H_13+qty
                            elif k==14:
                                H_14=H_14+qty
                            elif k==15:
                                H_15=H_15+qty
                            elif k==16:
                                H_16=H_16+qty
                            elif k==17:
                                H_17=H_17+qty
                            elif k==18:
                                H_18=H_18+qty
                            elif k==19:
                                H_19=H_19+qty
                            elif k==20:
                                H_20=H_20+qty
                            elif k==21:
                                H_21=H_21+qty
                            elif k==22:
                                H_22=H_22+qty
                            elif k==23:
                                H_23=H_23+qty
                    temp=pd.DataFrame([[terminal,H_00,H_01,H_02,H_03,H_04,H_05,H_06,H_07,H_08,H_09,H_10,H_11,H_12,H_13,H_14,H_15,H_16,H_17,H_18,H_19,H_20,H_21,H_22,H_23]],columns=BookingTrendcols)
                    D_bookingTerminal=D_bookingTerminal.append(temp)       
        
    #identifico il trend complessivo
    DailyWorkloadNetwork=D_bookingTerminal[['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
    DailyWorkloadNetwork=DailyWorkloadNetwork.sum(axis=0,skipna = True)
    
    fig1=plt.figure()
    plt.stem(DailyWorkloadNetwork)
    plt.title('Network Handling time windows')
    plt.ylabel('Total Container Handled')
    plt.xlabel('Daily timeline')
    
    outputfigure[f"productivity_workload_network"]=fig1
    #plt.close('all')
    
    #Traccio il profilo di workload per ogni terminal
    for i in range(0,len(Terminals)):
        
        terminal=Terminals[i]
        DailyWorkloadTerminal=D_bookingTerminal[D_bookingTerminal.Terminal==terminal]
        
        DailyWorkloadTerminal=DailyWorkloadTerminal[['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
        DailyWorkloadTerminal=DailyWorkloadTerminal.mean(axis=0,skipna = True)
        
        fig1=plt.figure()
        plt.stem(DailyWorkloadTerminal)
        plt.title('Terminal: '+str(terminal)+' Handling time windows')
        plt.ylabel('Average Quantity Handled per hour')
        plt.xlabel('Daily timeline')
        outputfigure[f"productivity_workload_{terminal}"]=fig1
        plt.close('all')
    return outputfigure, D_terminal, D_coverages 
