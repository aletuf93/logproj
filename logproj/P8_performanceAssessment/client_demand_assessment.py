# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from logproj.ml_explore import paretoChart
from logproj.P8_performanceAssessment.utilities_movements import getCoverageStats






def clientStatistics(D_mov,
                       clientfield='KLANT',
                       itemfamily='ContainerSize',
                       capacityfield='QUANTITY'):

    imageResult={}
    df_results=pd.DataFrame()

    accuracy, _ = getCoverageStats(D_mov,clientfield,capacityField='QUANTITY')
    D_OrderPerClient=D_mov.groupby([clientfield]).size().reset_index()
    D_OrderPerClient=D_OrderPerClient.rename(columns={0:'TotalOrders'})
    D_OrderPerClient=D_OrderPerClient.sort_values([clientfield])

    #creo pie-chart
    labels=D_OrderPerClient[clientfield]
    sizes=D_OrderPerClient.TotalOrders
    explode = 0.1*np.ones(len(sizes))

    fig1, ax1 = plt.subplots(figsize=(20,10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    #plt.title('Orders per client')
    #fig1.savefig(dirResults+'\\03-ClientDiagnosis_OrdersPerClient.png')
    imageResult['clients_pie']=fig1

    #Conto TEU e FEU per cliente
    D_movTypePerClient=D_mov.groupby([clientfield,itemfamily]).size().reset_index()
    D_movTypePerClient=D_movTypePerClient.rename(columns={0:'TotalContainer'})
    D_movTypePerClient=D_movTypePerClient.pivot(index=clientfield,columns=itemfamily,values='TotalContainer')

    #cols=['TEU','FEU','L5GO']
    #D_movTypePerClient=D_movTypePerClient[cols]

    D=pd.merge(D_movTypePerClient, D_OrderPerClient, left_on=[clientfield],right_on=[clientfield])
    D=D.fillna(0)

    #Fare un salvataggio finale della tabella per cliente
    df_results=D
    #aggiungo accuratess
    df_results['accuracy']=[accuracy for i in range(0,len(df_results))]

    #pareto sulle capacità prenotate per cliente
    D_capacityPerClient=D_mov.groupby([clientfield])[capacityfield].sum().reset_index()
    fig1=paretoChart(D_capacityPerClient,clientfield,capacityfield,'Pareto clients')


    imageResult['paretoClient']=fig1
    return imageResult, df_results



# %% da sistemare
     # costruisco pareto sui nodi visitati
def paretoNodeClient(D_mov,
                    clientfield='KLANT',
                    locationfromfield='LOADING_NODE',
                    locationtofield='DISCHARGING_NODE',
                    vehiclefield='VEHICLE_CODE',
                    capacityField='QUANTITY'
                    ):
    outputfigure={}
    output_df={}

    #se sono gli stessi campi non riesco a cumulare nulla
    if  (clientfield==locationfromfield) | (clientfield==locationtofield):
        print ("Same field for client and location from/to. Cannot proceed")
        return outputfigure,output_df
    for barge in set(D_mov[vehiclefield]):
         #print(barge)

         #filtro il dataframe
         D_clNode=D_mov[D_mov[vehiclefield]==barge]
         if len(D_clNode)>0:
             # calcolo le coperture
             accuracy, _ = getCoverageStats(D_clNode,[clientfield,locationfromfield,locationtofield,vehiclefield],
                                            capacityField=capacityField)



             D_clNode_from = pd.DataFrame(D_clNode.groupby([clientfield,locationtofield]).size()).reset_index()
             D_clNode_from = D_clNode_from.rename(columns={locationtofield:'Location'})

             D_clNode_to = pd.DataFrame(D_clNode.groupby([clientfield,locationfromfield]).size()).reset_index()
             D_clNode_to = D_clNode_to.rename(columns={locationfromfield:'Location'})

             D_clNode_all=pd.concat([D_clNode_from,D_clNode_to],axis=0)
             D_clNode_all=D_clNode_all.sort_values(by=0, ascending=False)
             D_clNode_all=D_clNode_all.dropna()
             D_clNode_all=D_clNode_all.reset_index(drop=True)

             #elimino le location già incontrate
             setLocation=[]
             for row in D_clNode_all.iterrows():
                 index=row[0]
                 rr=row[1]
                 if rr.Location.lower().strip() in setLocation:
                     D_clNode_all=D_clNode_all.drop(index)
                 else:
                     setLocation.append(rr.Location.lower().strip())


             #aggiungo nodi che non cumuano nulla per rappresentarli nella pareto
             D_clNode_all=D_clNode_all.groupby([clientfield])['Location'].nunique()
             D_clNode_all=pd.DataFrame(D_clNode_all)
             for client in set(D_clNode[clientfield]):
                 if client not in D_clNode_all.index.values:
                     #print(client)
                     temp=pd.DataFrame([0],index=[client],columns=['Location'])
                     D_clNode_all = pd.concat([D_clNode_all,temp])





             D_clNode_all=pd.DataFrame(D_clNode_all)
             D_clNode_all['Client']=D_clNode_all.index.values
             D_clNode_all['accuracy']= [accuracy for i in range(0,len(D_clNode_all))]



             titolo = f"BargeCode: {barge}"
             fig = paretoChart(D_clNode_all, 'Client', 'Location',titolo)
             outputfigure[f"pareto_vehicle_{barge}"]=fig
             output_df[f"pareto_vehicle_{barge}"]=D_clNode_all
    return outputfigure, output_df


# %% Violin chart (terminal vs clients)
def violinPlantTerminal(D_mov,plantField='LOADING_NODE',
                        clientField='DISCHARGING_NODE',
                        capacityField='QUANTITY'):
    # la funzione realizza un plot a violino per ogni nodo produttivo (plant)
    #indicando le quantita' movimentate verso ogni cliente

    output_figure = {}
    output_df={}



    accuracy, _ = getCoverageStats(D_mov,[clientField,plantField],capacityField=capacityField)
    df_out=pd.DataFrame([accuracy])

    D_clientTerminal = D_mov.groupby([plantField,clientField]).sum()[capacityField].reset_index()

    #f, ax = plt.subplots(figsize=(7, 7))
    #ax.set(yscale="log")
    fig=plt.figure()
    sns.violinplot(x=plantField, y=capacityField,
                        data=D_clientTerminal, palette="muted")
    output_figure['violin_plant_client']=fig
    output_df['violin_plant_client_coverages']=df_out

    return output_figure, output_df
