# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from logproj.DIST.globalBookingAnalysis import getAdvanceInPlanning, getCoverageStats
from logproj.ml_graphs import plotGraph
#%%

def createTabellaMovimenti( D_mov,
                            locfrom = 'LOADING_NODE',                     
                            locto= 'DISCHARGING_NODE',
                            capacityField='QUANTITY',
                            timeColumns={}
                            ):
    #Sdoppio ogni movimento in due righe una in e una out
    
    print("**** DEFINISCO D MOV IN/OUT ****")
    # verifico quali campi data sono presenti e identifico le colonne di raggruppamento
    columnsCompleteFrom = ['loadingpta', 'loadingptd', 'loadingata', 'loadingatd']
    columnsCompleteTo= ['dischargingpta','dischargingptd','dischargingata','dischargingatd']
    
    columnsPresentFrom = [ timeColumns[col] for col in list(timeColumns) if col in columnsCompleteFrom ]
    columnsPresentTo= [ timeColumns[col] for col in list(timeColumns) if col in columnsCompleteTo ]
    
    selectColumnFrom=list(D_mov.columns)
    for col in [locto, *columnsPresentTo]:
        if col in selectColumnFrom: selectColumnFrom.remove(col)
    
    selectColumnTo=list(D_mov.columns)
    for col in [locfrom, *columnsPresentFrom]:
        if col in selectColumnTo: selectColumnTo.remove(col)
    
    #identifico quali colonne sono presenti e come andarle a rinominare
    allcolumnstorename = {'loadingpta':'PTA',
                                 'loadingptd':'PTD',
                                 'loadingata':'ATA',
                                 'loadingatd':'ATD',
                                 'dischargingpta':'PTA',
                                 'dischargingptd':'PTD',
                                 'dischargingata':'ATA',
                                 'dischargingatd':'ATD',}
    
    renameDictionarycomplete =  {locto:'Location',
                                 locfrom:'Location'
            
                                 }
    for col in allcolumnstorename.keys():
        if col in timeColumns.keys():
            renameDictionarycomplete[timeColumns[col]]=allcolumnstorename[col]
     
    #sdoppio i movimenti e rinomino    
    D1=D_mov[selectColumnFrom]
    D1=D1.rename(columns=renameDictionarycomplete)
    D1['InOut']='IN'
    
    D2=D_mov[selectColumnTo]
    D2=D2.rename(columns=renameDictionarycomplete)
    D2['InOut']='OUT'
    
    
    #Creo la tabella dei movimenti
    D=pd.concat([D1,D2])
    
    #assegno quantita' e segni ai movimenti
    MovimentiIN=(D.InOut=='IN')*1
    MovimentiOUT=(D.InOut=='OUT')*(-1)
    D['Movementquantity']=MovimentiIN+MovimentiOUT
    D['Movementquantity']=D.Movementquantity*D[capacityField]
    
    return D

# %%
def defineRouteTable(D,agregationVariables=['VEHICLE_CODE','VOYAGE_CODE'],actual='PROVISIONAL'):
    #import a dataframe D containing movements and defines a route dataframe
    print("**** DEFINISCO ROUTE  ****")
    
    #costruisco un dizionario di aggregazione sul groupby
    #la capacita' verra' sommata
    #tutte le altre variabili diventeranno una lista
    
    #aggregationVariable
    
    aggregation_dictionary = {'Movementquantity':np.sum}
    if actual=='PROVISIONAL':
        listCol = [*agregationVariables,'Location','PTA','PTD','Movementquantity','_id']
    elif actual=='ACTUAL':
        listCol = [*agregationVariables,'Location','ATA','ATD','Movementquantity','_id']
    aggregation_columns = [col for col in D.columns if col not in listCol ]
    for col in aggregation_columns:
        aggregation_dictionary[col] = lambda group_series: list(set(group_series.tolist()))
        
    #rimuovo eventuali colonne contenenti dizionari
    #listKeys = aggregation_dictionary.keys()
    for col in list(aggregation_dictionary): 
        if any([ isinstance(i,dict) for i in D[col] ]):
            print(col)
            aggregation_dictionary.pop(col)
        
          
    
    #Ricostruisco la route effettuata
    if actual=='PROVISIONAL':
        D_route=D.groupby([*agregationVariables,'Location','PTA','PTD']).agg(aggregation_dictionary).reset_index()
        timestartfield='PTA'
        timeendfield='PTD'
        
       
    elif actual=='ACTUAL':
        D_route=D.groupby([*agregationVariables,'Location','ATA','ATD']).agg(aggregation_dictionary).reset_index()
        timestartfield='ATA'
        timeendfield='ATD'
    return D_route, timestartfield, timeendfield


# In[4]: #Statistiche sui voyage
def D_voyageStatistics(     D_mov,
                            timefield='TIMESTAMP_IN',
                            locfrom = 'LOADING_NODE',
                            locto= 'DISCHARGING_NODE',
                            timeColumns={},
                            capacityField='QUANTITY',
                            censoredData=False,
                            voyagefield ='VOYAGE_CODE',
                            actual='PROVISIONAL'):
    
    #ritorna due dataframe
    #D_route con tutti gli attributi basati sui singoli movimenti
    #D_arcs_route con tutti gli attributi basati sugli archi percorsi
    
    #inizializzo a vuoto
    D_route =  D_arcs_route  = D_coverages = pd.DataFrame()
    
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
    
    D_arcs_route=pd.DataFrame()
    
    
    D=createTabellaMovimenti(D_mov=D_mov,
                            locfrom = locfrom,
                            locto= locto,
                            capacityField=capacityField,
                            timeColumns=timeColumns)

    
             
    
    #ricostruisco le rotte
    D_route, timestartfield, timeendfield = defineRouteTable(D,agregationVariables =[voyagefield],actual=actual)
    
    #identifico i possibili viaggi
    Voyages=np.unique(D_route[voyagefield])
    
    #verifico se vi siano dei dati censurati da segnalare
    
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
            D_route=D_route[(D_route[timestartfield]>pd.to_datetime(lowerBoundDataCensored)) & (D_route[timeendfield]<pd.to_datetime(lastPlanningDay))]
            D_route = D_route.reset_index(drop=True)
    
    #proseguo solo se ho dati non censurati
    if len(D_route)==0:
        D_route=pd.DataFrame(["No uncensored data"])
        return D_route,  D_arcs_route, D_coverages                                         
    D_route['inventory']=np.nan
    
    print("**** RICOSTRUZIONE DEGLI INVENTARI  ****")
    #scorro sui singoli viaggi e vado a calcolare la capacita' residua
    for i in range(0,len(Voyages)):
        #i=0
        voyage=Voyages[i]
        route =  D_route[D_route[voyagefield]==voyage]
        
        print(f"==RICOSTRUISCO INVENTARIO VIAGGIO {voyage}, con {len(route)} MOVIMENTI")
        #se ho una rotta
        if len(route)>0:
        
            #ordino in base al tempo
            route=route.sort_values([timeendfield])
            
            #Identifico la capacità dai dati
            #vessel=np.unique(route[vehiclefield])[0] # c'e' un solo veicolo associato ad ogni viaggio
            #capDati=np.double(capBarges[capBarges[vehiclefield]==vessel][capfield])
            
            
            
            
            #definisco i movimenti pianificati sulla chiatta
            counter=0
            allIndex=[] #creo una lista di indici che poi uso per aggiornare solo questa porzione di tabella
            for index, row in route.iterrows(): #gli indici di route sono gli stessi in D_route
                if counter==0:
                    D_route['inventory'].loc[index]=row['Movementquantity']
                    allIndex.append(index)
                    
                else:
                    D_route['inventory'].loc[index]=row['Movementquantity'] + D_route['inventory'].loc[allIndex[counter-1]]
                    allIndex.append(index)
                counter=counter+1
            
            #Calcolo la stima della capacità andando a portare tutto sopra lo zero
            allCapacities = D_route[D_route[voyagefield]==voyage]['inventory']
            slack=np.double(-min(allCapacities))
            D_route['inventory'].loc[allIndex]=D_route[D_route[voyagefield]==voyage]['inventory']+slack
            capMax=max(D_route['inventory'].loc[allIndex])
            
            #riassegno il valore di route alla tabella aggiornata con i valori di inventario e riordino
            route=D_route[D_route[voyagefield]==voyage]
            route=route.sort_values([timeendfield])
            
            #scorro la rotta per creare il dataframe dei movimenti from-to
            for k in range(0,len(route)-1):
                #k=0           
                
                #identifico il movimento corrente e il successivo
                currentMovement=route.iloc[k]
                nextMovement=route.iloc[k+1]
    
                rowDictionary={'arcFrom':currentMovement.Location,
                               'arcTo': nextMovement.Location,
                               'departureFromALAP': currentMovement[timeendfield],
                               'arrivalToASAP':nextMovement[timestartfield],
                               'inventory':currentMovement.inventory,
                               'capacity':capMax-currentMovement.inventory}
                #appendo tutte le altre del from
                add_columns_from = [col for col in currentMovement.index if col not in ['Location','timeendfield','inventory'] ]
                for col in add_columns_from:
                    rowDictionary[f"{col}_from"] = currentMovement[col]
                
                #appendo tutte le altre del to
                
                add_columns_to = [col for col in nextMovement.index if col not in ['Location','timestartfield','inventory'] ]
                for col in add_columns_to:
                    rowDictionary[f"{col}_to"] = nextMovement[col]
                    
                #aggiungo al dataframe dei risultati
                D_arcs_route = D_arcs_route.append(pd.DataFrame([rowDictionary]))
    
    return D_route,  D_arcs_route, D_coverages                   
                          



# %%
def returnFigureVoyage(D_route, D_arcs_route, lastPlanningDay=[], lowerBoundDataCensored=[], filteringfield='VOYAGE_CODE',sortTimefield='PTD'):        
    
    #ritorna un dizionario di figure di inventario e connessioni su grafo per ogni codice di viaggio
    
    figure_results={}
    for voyage in set(D_route[filteringfield]):
        #voyage='Vessel 10'
        #genero grafici di inventario
        D_plannedRouteVessel =  D_route[D_route[filteringfield]==voyage]
        
        
        if len(D_plannedRouteVessel)>0:
            D_plannedRouteVessel = D_plannedRouteVessel.sort_values(by=sortTimefield)
            
            #Genero grafici di capacità
            figure=plt.figure(figsize=(20,10))
            plt.step(D_plannedRouteVessel[sortTimefield],D_plannedRouteVessel['inventory'], color='orange')
            plt.title(str(voyage)+' inventory')
            plt.xticks(rotation=30)
            
            #traccio la capacità
            #if (len(lastPlanningDay)>0) & (len(lowerBoundDataCensored)>0):
            capMax=max(D_plannedRouteVessel['inventory'])
            plt.plot(D_plannedRouteVessel[sortTimefield],[capMax]*len(D_plannedRouteVessel),'r--')
            plt.axvline(x=lastPlanningDay, color='red',linestyle='--')
            plt.axvline(x=lowerBoundDataCensored, color='red',linestyle='--')
            figure_results[f"{filteringfield}_{voyage}_inventory"]=figure
        
        #plt.close('all')
                
        # genero grafici di flusso su grafo     
        D_plannedRouteVessel_fromTo =  D_arcs_route[D_arcs_route[f"{filteringfield}_from"]==voyage]
        
        if len(D_plannedRouteVessel_fromTo)>0:
            #traccio le route su grafo
            FlowAnalysis=D_plannedRouteVessel_fromTo.groupby(['arcFrom', 'arcTo']).size().reset_index() 
            FlowAnalysis=FlowAnalysis.rename(columns={0:'Trips'})
            
            fig1=plotGraph(df=FlowAnalysis,
                           edgeFrom='arcFrom',
                           edgeTo='arcTo',
                           distance='Trips',
                           weight='Trips',
                           title=str(voyage),
                           arcLabel=True)
            figure_results[f"{filteringfield}_{voyage}_graph"]=fig1
     
                    
    return figure_results


# %% network graph clock
def graphClock(D_mov,
               loadingNode='LOADING_NODE',
               dischargingNode='DISCHARGING_NODE',
               sortingField='PTA_FROM',
               vehicle='VEHICLE_CODE',
               capacityField='QUANTITY',
               timeColumns={},
               actual='PROVISIONAL'):
    
    output_figure={}
    output_df={}
    #identifico colonne necessarie e calcolo coperture
    
    if actual=='PROVISIONAL':
        colonneNecessarie = ['loadingptd','dischargingpta']
        if all([column in timeColumns.keys()  for column in colonneNecessarie ]):
            allcolumns = [loadingNode,dischargingNode,vehicle, timeColumns['loadingptd'],timeColumns['dischargingpta']]
            accuracy, _ = getCoverageStats(D_mov,analysisFieldList=allcolumns,capacityField='QUANTITY')
        else:
            colonneMancanti=[column   for column in  colonneNecessarie if column not in timeColumns.keys()]
            D_coverages=pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
    elif actual == 'ACTUAL':
        colonneNecessarie = ['loadingatd','dischargingata']
        if all([column in timeColumns.keys()  for column in colonneNecessarie ]):
            allcolumns = [locfrom,locto,vehicle, timeColumns['loadingatd'],timeColumns['dischargingata']]
            accuracy, _ = getCoverageStats(D_mov,analysisFieldList=allcolumns,capacityField='QUANTITY')
    output_df[f"coverages_{actual}"]=pd.DataFrame(accuracy)        
    #identifico tutti i terminal e assegno un'ordinata
    #andrebbero ordinati su una retta
    
    terminal_dict={}
    D_mov=D_mov.sort_values(by=[sortingField])
    terminals = list(set([*D_mov[loadingNode], *D_mov[dischargingNode]]))
    for i in range(0,len(terminals)):
        terminal_dict[terminals[i]]=i
        
    #identifico i movimenti
    
    D = createTabellaMovimenti( D_mov,
                            locfrom = loadingNode,                     
                            locto= dischargingNode,
                            capacityField=capacityField,
                            timeColumns=timeColumns
                            )
    D_route, timestartfield, timeendfield = defineRouteTable(D,
                                                        agregationVariables =[vehicle],
                                                             actual=actual)
    
        
    for vessel in set(D_route[vehicle]):
        D_mov_filtered = D_route[D_mov[vehicle]==vessel]
        D_mov_filtered = D_route.sort_values(by=timestartfield)
        D_mov_filtered=D_mov_filtered.dropna(subset=[timestartfield, 'Location'])
        
        #realizzo grafico su tutto l'asse temporale
        fig1=plt.figure()
        for i in range(1,len(D_mov_filtered)):
            
            #plotto viaggio
            x_array = [D_mov_filtered[timeendfield].iloc[i-1], D_mov_filtered[timestartfield].iloc[i]]
            y_array = [terminal_dict[D_mov_filtered['Location'].iloc[i-1]], terminal_dict[D_mov_filtered['Location'].iloc[i]]]
            plt.plot(x_array,y_array,color='orange',marker='o')   
            
            #plotto attesa
            x_array = [D_mov_filtered[timestartfield].iloc[i], D_mov_filtered[timeendfield].iloc[i]]
            y_array = [terminal_dict[D_mov_filtered['Location'].iloc[i]], terminal_dict[D_mov_filtered['Location'].iloc[i]]]
            plt.plot(x_array,y_array,color='orange',marker='o')  
        plt.ylabel('Terminal')
        plt.xlabel('Time')
        output_figure[f"Train_chart_{vessel}_{actual}"] = fig1
        
        if actual=='PROVISIONAL':
            time_from  = timeColumns['loadingptd']
            time_to  = timeColumns['dischargingpta']
        elif actual=='ACTUAL':
            time_from  = timeColumns['loadingatd']
            time_to  = timeColumns['dischargingata']
            
    
        #realizzo grafico raggruppando su un giorno
        D_train=D_mov[D_mov[vehicle]==vessel]
        D_train['hour_from']=D_train[time_from].dt.time
        D_train['hour_to']=D_train[time_to].dt.time 
        D_graph=D_train.groupby([loadingNode,dischargingNode,'hour_from','hour_to']).sum()[capacityField].reset_index()            
        D_graph=D_graph.sort_values(by=capacityField, ascending=False)
        fig1=plt.figure()
        for i in range(0,len(D_graph)):
            x_array = [D_graph['hour_from'].iloc[i], D_graph['hour_to'].iloc[i]]
            y_array = [terminal_dict[D_graph[loadingNode].iloc[i]], terminal_dict[D_graph[dischargingNode].iloc[i]]]
            
            my_day = datetime.date(1990, 1, 1)
            x_array = [ datetime.datetime.combine(my_day, t) for t in x_array ]
            
            plt.title(f"Train schedule chart VEHICLE: {vessel}")
            plt.plot(x_array,y_array,color='orange',marker='o',linewidth =  np.log(D_graph[capacityField].iloc[i]))
        output_figure[f"Train_chart_daily_{vessel}_{actual}"] = fig1
        plt.close('all')
    return output_figure, output_df