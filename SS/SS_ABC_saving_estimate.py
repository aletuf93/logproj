# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:07:31 2019

@author: Alessandro
"""
# In[1]: importo pacchetti
#from database.back_db_login_manager import setConnection

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def filterD_pop(D_pop, listVehicle, listSubarea, listIdwh, nodeCode):
    esito=True
    errore=""

    ####################### get orderlist ######################################
    D_pop_filtered=D_pop

    #filter by nodecode
    D_pop_filtered=D_pop_filtered[D_pop_filtered['NODECODE'].isin([nodeCode])]
    if len(D_pop_filtered)==0: # no SKU descriptions in this nodecode
        esito=False
        errore=f'There are no movements associated to the node code: {nodeCode}'
        return esito, errore, []

    #filter by idwh
    if len(listIdwh)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['IDWH'].isin(listIdwh)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this Idwh
            esito=False
            errore=f'There are no movements associated to the selected idwh'
            return esito, errore, []

    #filter by subarea
    if len(listSubarea)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['WHSUBAREA'].isin(listSubarea)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this whsubarea
            esito=False
            errore=f'There are no movements associated to the selected logical warehouses (whsubarea)'
            return esito, errore, []

    #filter by vehicle
    if len(listVehicle)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['VEHICLECATEGORY'].isin(listVehicle)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this vehicle
            esito=False
            errore=f'There are no movements associated to the selected vehicle'
            return esito, errore, []

    D_pop_filtered=D_pop_filtered.reset_index()

    return esito,errore,D_pop_filtered


def filterD_popD_WH(D_pop,D_WH, listVehicle, listSubarea, listIdwh, nodeCode):
    esito=True
    errore=""

    ####################### get orderlist ######################################
    D_pop_filtered=D_pop

    #filter by nodecode
    D_pop_filtered=D_pop_filtered[D_pop_filtered['NODECODE'].isin([nodeCode])]
    if len(D_pop_filtered)==0: # no SKU descriptions in this nodecode
        esito=False
        errore=f'There are no movements associated to the node code: {nodeCode}'
        return esito, errore, [],[],[]

    #filter by idwh
    if len(listIdwh)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['IDWH'].isin(listIdwh)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this Idwh
            esito=False
            errore=f'There are no movements associated to the selected idwh'
            return esito, errore, [],[],[]

    #filter by subarea
    if len(listSubarea)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['WHSUBAREA'].isin(listSubarea)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this whsubarea
            esito=False
            errore=f'There are no movements associated to the selected logical warehouses (whsubarea)'
            return esito, errore, [],[],[]

    #filter by vehicle
    if len(listVehicle)>0:
        D_pop_filtered=D_pop_filtered[D_pop_filtered['VEHICLECATEGORY'].isin(listVehicle)]
        if len(D_pop_filtered)==0: # no SKU descriptions in this vehicle
            esito=False
            errore=f'There are no movements associated to the selected vehicle'
            return esito, errore, [],[],[]

    D_pop_filtered=D_pop_filtered.reset_index()

    ########################## get P and Q #############################################à
    D_WH_filtered=D_WH

    #filter by nodecode
    D_WH_filtered=D_WH_filtered[D_WH_filtered['NODECODE'].isin([nodeCode])]
    if len(D_WH_filtered)==0: # no SKU descriptions in this nodecode
        esito=False
        errore=f'There are no coordinates (locations table) associated to the node code: {nodeCode}'
        return esito, errore, [],[],[]

    #filter by idwh
    if len(listIdwh)>0:
        D_WH_filtered=D_WH_filtered[D_WH_filtered['IDWH'].isin(listIdwh)]
        if len(D_WH_filtered)==0: # no SKU descriptions in this Idwh
            esito=False
            errore=f'There are no coordinates (locations table) associated to the selected idwh'
            return esito, errore, [],[],[]

    #filter by subarea
    if len(listSubarea)>0:
        D_WH_filtered=D_WH_filtered[D_WH_filtered['WHSUBAREA'].isin(listSubarea)]
        if len(D_WH_filtered)==0: # no SKU descriptions in this whsubarea
            esito=False
            errore=f'There are no coordinates (locations table) associated to the selected logical warehouses (whsubarea)'
            return esito, errore, [],[],[]

    #filter by vehicle
    if len(listVehicle)>0:
        D_WH_filtered=D_WH_filtered[D_WH_filtered['VEHICLECATEGORY'].isin(listVehicle)]
        if len(D_WH_filtered)==0: # no SKU descriptions in this vehicle
            esito=False
            errore=f'There are no coordinates (locations table) associated to the selected vehicle'
            return esito, errore, [],[],[]

    D_WH_filtered=D_WH_filtered.reset_index()
    P=np.nanmax(D_WH_filtered.P)
    Q=np.nanmax(D_WH_filtered.Q)
    return esito,errore,D_pop_filtered,P,Q

# In[1]:
def calculateABCsaving(p,q,D_pop_filtered):

    ############################# SCENARIO RANDOM #################################
    if (~np.isnan(p) or ~np.isnan(q)):


        #Conto pick in e out
        pickIN=sum((D_pop_filtered[D_pop_filtered['INOUT']=='+']).popularity)
        pickOUT=sum((D_pop_filtered[D_pop_filtered['INOUT']=='-']).popularity)


        #Random scenario

        #I/O distribuito sul fronte
        r_cicloSemplice=(q/2+p/3)*2



        KmIn_rand=pickIN*r_cicloSemplice
        KmOut_rand=pickOUT*r_cicloSemplice


        SAVING_IN=[]
        SAVING_OUT=[]
        soglieA=[]
        soglieB=[]

        ############################# SCENARIO ABC ###################################
        for i in range(0,100,10):
            for j in range(i+1,100,10):
                sogliaA=i/100
                sogliaB=j/100

                D_pop_totale=D_pop_filtered.groupby('ITEMCODE')['popularity'].sum().reset_index()
                D_pop_totale=D_pop_totale.sort_values(by='popularity',ascending=False)
                
               
                sogliaClasseA=int(np.round(sogliaA*len(D_pop_totale)))
                sogliaClasseB=int(np.round(sogliaB*len(D_pop_totale)))

                

                ITEM_A=D_pop_totale['ITEMCODE'].iloc[0:sogliaClasseA].reset_index()
                ITEM_B=D_pop_totale['ITEMCODE'].iloc[sogliaClasseA:sogliaClasseB].reset_index()
                ITEM_C=D_pop_totale['ITEMCODE'].iloc[sogliaClasseB:len(D_pop_filtered)].reset_index()

                #Estraggo le pickIN
                pickIN_A=D_pop_filtered[(D_pop_filtered['INOUT']=='+') & (D_pop_filtered['ITEMCODE'].isin(ITEM_A['ITEMCODE']))]
                pickIN_B=D_pop_filtered[(D_pop_filtered['INOUT']=='+') & (D_pop_filtered['ITEMCODE'].isin(ITEM_B['ITEMCODE']))]
                pickIN_C=D_pop_filtered[(D_pop_filtered['INOUT']=='+') & (D_pop_filtered['ITEMCODE'].isin(ITEM_C['ITEMCODE']))]

                num_pickin_A=sum(pickIN_A.popularity)
                num_pickin_B=sum(pickIN_B.popularity)
                num_pickin_C=sum(pickIN_C.popularity)

                #Estraggo le pickOUT
                pickOUT_A=D_pop_filtered[(D_pop_filtered['INOUT']=='-') & (D_pop_filtered['ITEMCODE'].isin(ITEM_A['ITEMCODE']))]
                pickOUT_B=D_pop_filtered[(D_pop_filtered['INOUT']=='-') & (D_pop_filtered['ITEMCODE'].isin(ITEM_B['ITEMCODE']))]
                pickOUT_C=D_pop_filtered[(D_pop_filtered['INOUT']=='-') & (D_pop_filtered['ITEMCODE'].isin(ITEM_C['ITEMCODE']))]

                num_pickout_A=sum(pickOUT_A.popularity)
                num_pickout_B=sum(pickOUT_B.popularity)
                num_pickout_C=sum(pickOUT_C.popularity)

                len_q_A=len(ITEM_A)/( len(ITEM_A) + len(ITEM_B) + len(ITEM_C) )
                len_q_B=len(ITEM_B)/( len(ITEM_A) + len(ITEM_B) + len(ITEM_C) )
                len_q_C=len(ITEM_C)/( len(ITEM_A) + len(ITEM_B) + len(ITEM_C) )


                #Calcolo i km
                #check OK numero picks
                if((num_pickin_A+num_pickin_B+num_pickin_C)==pickIN) & ((num_pickout_A+num_pickout_B+num_pickout_C)==pickOUT):

                    #I/O distribuito sul fronte
                    dist_A=(q*len_q_A/2 + p/3 )*2
                    dist_B=(q*(len_q_A+len_q_B/2) + p/3 )*2
                    dist_C=(q*(len_q_A+len_q_B+len_q_C/2) +p/3)*2

                    KmIn_ABC=(num_pickin_A*dist_A + num_pickin_B*dist_B + num_pickin_C*dist_C)
                    KmOut_ABC=(num_pickout_A*dist_A + num_pickout_B*dist_B + num_pickout_C*dist_C)

                    if (KmIn_rand==0): #avoid division by zero
                        sav_IN=0
                    else:
                        sav_IN=1 - float(KmIn_ABC/KmIn_rand)

                    if (KmOut_rand==0): #avoid division by zero
                        sav_OUT=0
                    else:
                        sav_OUT=1 - float(KmOut_ABC/KmOut_rand)

                    SAVING_IN.append(sav_IN)
                    SAVING_OUT.append(sav_OUT)
                    soglieA.append(sogliaA)
                    soglieB.append(sogliaB)

                    #calcolo il miglior scenario di saving
                    SAV_TOT = np.asarray(SAVING_IN) + np.asarray(SAVING_OUT)
                    idx = np.nanargmax(SAV_TOT)
                    best_A=np.round(soglieA[idx],1)
                    best_B=np.round(soglieB[idx],1)

                    #assign ABC to SKUs
                    D_pop_filtered=D_pop_filtered.sort_values(by='popularity',ascending=False)
                    D_pop_ABC = D_pop_filtered.groupby('ITEMCODE')['popularity'].sum().reset_index()
                    D_pop_ABC=D_pop_ABC.sort_values(by='popularity',ascending=False)
                    D_pop_ABC['Class']=''
                    num_a=int(best_A*len(D_pop_ABC))
                    num_b=int(best_B*len(D_pop_ABC))
                    D_pop_ABC.Class.iloc[0:num_a]='A'
                    D_pop_ABC.Class.iloc[num_a:num_b]='B'
                    D_pop_ABC.Class.iloc[num_b:len(D_pop_ABC)]='C'

                else:
                    print("Error: num pick scenario ABC does not match num pick scenario random")

        return soglieA, soglieB, SAVING_IN, SAVING_OUT,best_A,best_B, SAV_TOT[idx], D_pop_ABC
    else:
        return [],[],[],[]


   

    '''
    ################################# DEBUG OFFLINE ###########################
    #import from a level above
    import sys
    sys.path.append('../..')
    
    #import packages
    import database.back_db_queries as qq
    
    #other import
    import plotly.graph_objs as go
    from plotly.offline import plot
    
    caseStudy=1
    listVehicle=["FORKLIFT"]
    listSubarea=[]
    listIdwh=[]
    nodeCode="93"

    #represent layout
    D_WH=qq.importWarehousePQ(caseStudy)
    D_pop=qq.importWarehousePopularity(caseStudy)
    esito,errore,D_pop_filtered,p,q=filterD_popD_WH(D_pop,D_WH, listVehicle, listSubarea, listIdwh, nodeCode)
    soglieA, soglieB, SAVING_IN, SAVING_OUT,best_A,best_B, SAV_TOT, D_pop_ABC = calculateABCsaving(p,q,D_pop_filtered)
    
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d


    index_max_in = np.argmax(SAVING_IN)
    index_max_out = np.argmax(SAVING_OUT)
    print(f"sogliaA_IN:{soglieA[index_max_in]}")
    print(f"sogliaB_IN:{soglieB[index_max_in]}")

    print(f"sogliaA_OUT:{soglieA[index_max_out]}")
    print(f"sogliaB_OUT:{soglieB[index_max_out]}")


    #genero grafico IN
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(soglieA, soglieB, SAVING_IN)
    ax.set_xlabel('soglia A')
    ax.set_ylabel('soglia B')
    ax.set_zlabel('Saving')
    ax.set_title('Saving IN')
    fig.show()

    #genero grafico OUT
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(soglieA, soglieB, SAVING_OUT)
    ax.set_xlabel('soglia A')
    ax.set_ylabel('soglia B')
    ax.set_zlabel('Saving')
    ax.set_title('Saving OUT')
    fig.show()
    '''
