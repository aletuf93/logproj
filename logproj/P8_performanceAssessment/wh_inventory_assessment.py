
#%% import packages from other folders

import pandas as pd
import numpy as np

#from scipy.stats import poisson

import matplotlib.pyplot as plt
#import database.mongo_queries as qq
#import database.models.odm_logistics_mongo as model_log


import logproj.stat_time_series as ts
#from logproj.P1_familyProblem.part_classification import returnsparePartclassification
from logproj.information_framework import returnInventoryPart
from logproj.information_framework import returnProbabilisticInventory

from logproj.ml_explore import paretoDataframe


# %% POPULARITY INDEX
def calculatePopularity(movements):
    '''


    Parameters
    ----------
    movements : TYPE pandas series
        DESCRIPTION. the series of the movement with one item per day

    Returns
    -------
    pop_in : TYPE float
        DESCRIPTION. popularity IN per day
    pop_out : TYPE
        DESCRIPTION. popularity OUT per day

    '''

    pop_in = len(movements[movements>0])/len(movements)
    pop_out = len(movements[movements<0])/len(movements)
    pop_absolute_in = len(movements[movements>0])
    pop_absolute_out = len(movements[movements<0])
    return pop_in, pop_out, pop_absolute_in, pop_absolute_out

# %% COI INDEX
def calculateCOI(inventory):
    '''


    Parameters
    ----------
    inventory : TYPE list
        DESCRIPTION. list of inventory values

    Returns
    -------
    COI_in : TYPE float
        DESCRIPTION. daily COI index IN
    COI_out : TYPE float
        DESCRIPTION. daily COI index OUT

    '''

    #define inventory from movements
    movements = movementfunctionfromInventory(inventory)
    movements=movements.dropna()
    pop_in, pop_out,_,_ = calculatePopularity(movements['QUANTITY'])

    #calculate daily COI
    I_t_avg = np.nanmean(inventory)
    if I_t_avg>0:
        COI_in = pop_in/I_t_avg
        COI_out = pop_out/I_t_avg
    else:
        COI_in=COI_out=np.nan

    return COI_in, COI_out

# %% TURN INDEX
def calculateTurn(inventory):
    '''


    Parameters
    ----------
    inventory : TYPE list
        DESCRIPTION. list of inventory values

    Returns
    -------
    turn : TYPE float
        DESCRIPTION. daily TURN index

    '''
    #define inventory from movements
    movements = movementfunctionfromInventory(inventory)
    movements=movements.dropna()

    #calculate the average outbound quantity per day
    out_qty_day = -np.sum(movements[movements['QUANTITY']<0]['QUANTITY'])/len(movements)

    #calculate average inventory quantity
    I_t_avg = np.nanmean(inventory)
    if I_t_avg>0:
        turn =out_qty_day/I_t_avg
    else:
        turn=np.nan


    return turn


# %% ORDER COMPLETION INDEX
def calculateOrderCompletion(D_mov, itemcode, itemfield='ITEMCODE', ordercodefield='ORDERCODE'):
    '''


    Parameters
    ----------
    D_mov : TYPE pandas dataframe
        DESCRIPTION. dataframe with movements reporting ordercode and itemcode columns
    itemcode : TYPE string
        DESCRIPTION. itemcode to calculate the order competion (OC) index
    itemfield : TYPE, optional string name of D_mov clumn with itemcode
        DESCRIPTION. The default is 'ITEMCODE'.
    ordercodefield : TYPE, optional string name of D_mov clumn with ordercode
        DESCRIPTION. The default is 'ORDERCODE'.

    Returns
    -------
    OC : TYPE float
        DESCRIPTION. order completion (OC) index of SKU itemcode

    '''
    #clean data
    D_mov=D_mov[[itemfield,ordercodefield]]
    D_mov=D_mov[D_mov[ordercodefield]!='nan']
    D_mov=D_mov.dropna()
    D_mov=D_mov.reset_index()



    orders = list(set(D_mov[D_mov[itemfield]==itemcode][ordercodefield]))

    idx = [j in orders for j in D_mov[ordercodefield]]
    D_orders = D_mov.loc[idx]

    OC = 0
    for ordercode in orders:
        D_orders_filtered = D_orders[D_orders[ordercodefield]==ordercode]
        OC=OC+1/len(D_orders_filtered)
    return OC

def fourierAnalysisInventory(inventory):
    '''

    Parameters
    ----------
    I_t : TYPE TYPE list
        DESCRIPTION. list of inventory values
    Returns
    -------
    first_carrier : TYPE float
        DESCRIPTION. frequency (in 1/days) with the highest amplitude value
    period : TYPE float
        DESCRIPTION. period (in days) associated with the frequency with the highest amplitude value
    '''
    D = ts.fourierAnalysis(np.array(inventory))
    D=D.sort_values(by='Amplitude', ascending=False)
    first_carrier = D.iloc[0]['Frequency_domain_value'] #1/days
    period = 1/first_carrier
    return first_carrier, period

# %%

def movementfunctionfromInventory(I_t_cleaned):
    '''


    Parameters
    ----------
    I_t_cleaned : TYPE list
        DESCRIPTION. list of inventory values without nan values

    Returns
    -------
    M_t : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe of movements

    '''
    M_t=[]
    for j in range(1,len(I_t_cleaned)):
        M_t.append(I_t_cleaned[j]-I_t_cleaned[j-1])
    M_t=pd.DataFrame(M_t,columns=['QUANTITY'])
    return M_t

# %%

def assessInterarrivalTime(I_t):
    #remove nan values
    I_t_cleaned = [x for x in I_t if str(x) != 'nan'] #remove nan inventories (e.g. at the beginning of the series if the part is not in the WH)

    #generate the movement function
    M_t = movementfunctionfromInventory(I_t_cleaned)

    M_t_in=M_t[M_t['QUANTITY']>0]
    interarrival_time = []
    for j in range(1,len(M_t_in)):
        interarrival_time.append(M_t_in.index[j] - M_t_in.index[j-1])

    mean_interarrival_time_in = np.mean(interarrival_time)
    std_interarrival_time_in = np.std(interarrival_time)
    return mean_interarrival_time_in, std_interarrival_time_in, interarrival_time
   

# %%
def updatePartInventory(D_SKUs,D_movements,D_inventory,timecolumn_mov,itemcodeColumns_sku,itemcodeColumns_mov,itemcodeColumns_inv):
    
    D_SKUs['INVENTORY_QTY'] = [[] for i in range(0,len(D_SKUs))]
    
    #  define the inventory quantity
    firstDay = min(D_movements[timecolumn_mov]).date()
    lastDay = max(D_movements[timecolumn_mov]).date()
    timeLine = pd.date_range(start=firstDay, end=lastDay).to_frame()
    timeLineDays= ts.sampleTimeSeries(timeLine[0],'day').to_frame()
    timeLineDays.columns=['TIMELINE']
   
    D_SKUs = D_SKUs.reset_index(drop=True)
    #  Build a daily inventory array for each part
    for index, row in D_SKUs.iterrows():
        #print(part)
        #part = list(set(D_mov['ITEMCODE']))[0]
        part = row[itemcodeColumns_sku]

        #filter movements by itemcode
        D_mov_part = D_movements[D_movements[itemcodeColumns_mov]==part]

        #filter inventory by itemcode
        D_inv_part = D_inventory[D_inventory[itemcodeColumns_inv]==part]

        array_days, array_inventory = returnInventoryPart(D_mov_part, D_inv_part, timeLineDays)
        #plt.plot(array_days,array_inventory)

        #update the dataframe
        D_SKUs.at[index,'INVENTORY_QTY'] = array_inventory
    return D_SKUs


# %% UPDATE POPULARITY INDEX
def updatePopularity(D_SKUs):
    
    #create result columns
    D_SKUs['POP_IN']=np.nan
    D_SKUs['POP_OUT']=np.nan
    D_SKUs['POP_IN_TOT']=np.nan
    D_SKUs['POP_OUT_TOT']=np.nan
    
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            POP_IN, POP_OUT, POP_IN_TOT, POP_OUT_TOT = calculatePopularity(movements['QUANTITY'])
            
            #update the dataframe
            D_SKUs.at[index,'POP_IN']=POP_IN
            D_SKUs.at[index,'POP_OUT']=POP_OUT
            D_SKUs.at[index,'POP_IN_TOT']=POP_IN_TOT
            D_SKUs.at[index,'POP_OUT_TOT']=POP_OUT_TOT
    return D_SKUs

# %% UPDATE COI INDEX
def updateCOI(D_SKUs):
    
    #create result columns
    D_SKUs['COI_IN']=np.nan
    D_SKUs['COI_OUT']=np.nan
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            COI_IN, COI_OUT = calculateCOI(I_t)
            
            #update the dataframe
            D_SKUs.at[index,'COI_IN']=COI_IN
            D_SKUs.at[index,'COI_OUT']=COI_OUT
            
    return D_SKUs

# %% UPDATE TURN INDEX
def updateTURN(D_SKUs):
    
    #create result columns
    D_SKUs['TURN']=np.nan
    
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            TURN = calculateTurn(I_t)
            
            #update the dataframe
            D_SKUs.at[index,'TURN']=TURN
            
            
    return D_SKUs


# %% UPDATE OC INDEX
def updateOrderCompletion(D_SKUs, D_mov):
    
    
    
    #create result columns
    D_SKUs['OC']=np.nan
    
    
    for index, row in D_SKUs.iterrows():
        
        part = row['ITEMCODE']
        
        #calculate the popularity
        OC = calculateOrderCompletion(D_mov, part, itemfield='ITEMCODE', ordercodefield='ORDERCODE')
        
        
        #update the dataframe
        D_SKUs.at[index,'OC']=OC
            
    return D_SKUs


# %% UPDATE INVENTORY PARAMETERS

def updateInventoryParams(D_SKUs):
    
    #create result columns
    D_SKUs['INVENTORY_REAL_MIN']=np.nan
    D_SKUs['INVENTORY_REAL_MAX']=np.nan
    D_SKUs['INVENTORY_REAL_AVG']=np.nan
    D_SKUs['INVENTORY_REAL_STD']=np.nan
    D_SKUs['INVENTORY_PROB_MIN']=np.nan
    D_SKUs['INVENTORY_PROB_MAX']=np.nan
    D_SKUs['INVENTORY_PROB_AVG']=np.nan
    D_SKUs['INVENTORY_PROB_STD']=np.nan
    
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            min_real_I_t, max_real_I_t, avg_real_I_t, std_real_I_t, min_probabilistic_I_t, max_probabilistic_I_t, avg_probabilistic_I_t, std_probabilistic_I_t = returnProbabilisticInventory(I_t)
            
            #update the dataframe
            
            D_SKUs.at[index,'INVENTORY_REAL_MIN']=min_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_MAX']=max_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_AVG']=avg_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_STD']=std_real_I_t
            D_SKUs.at[index,'INVENTORY_PROB_MIN']=min_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_MAX']=max_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_AVG']=avg_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_STD']=std_probabilistic_I_t
            
            
    return D_SKUs


# %% UPDATE INTERARRIVAL TIME

def updateInterarrivalTime(D_SKUs):
    
    #create result columns
    D_SKUs['INTERARRIVAL_MEAN_IN']=np.nan
    D_SKUs['INTERARRIVAL_STD_IN']=np.nan
    
    
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            mean_interarrival_time_in, std_interarrival_time_in, _ = assessInterarrivalTime(I_t)
            
            #update the dataframe
            
            D_SKUs.at[index,'INTERARRIVAL_MEAN_IN']=mean_interarrival_time_in
            D_SKUs.at[index,'INTERARRIVAL_STD_IN']=std_interarrival_time_in
       
    return D_SKUs


# %% UPDATE FOURIER ANALYSIS

def updateFourieranalysis(D_SKUs):
    
    #create result columns
    D_SKUs['FOURIER_CARRIER']=np.nan
    D_SKUs['FOURIER_PERIOD']=np.nan
    
    
    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            carrier, period = fourierAnalysisInventory(I_t)
            
            #update the dataframe
            
            D_SKUs.at[index,'FOURIER_CARRIER']=carrier
            D_SKUs.at[index,'FOURIER_PERIOD']=period
       
    return D_SKUs



# %% PARETO AND HISTOGRAM PLOT



def whIndexParetoPlot(D_SKUs,columnIndex):
    
    
    output_figures = {}
    
    
    #define the pareto values
    D_SKUs_pop = paretoDataframe(D_SKUs,columnIndex)
    
    #build the pareto figures
    fig1 = plt.figure()
    plt.plot(np.arange(0,len(D_SKUs_pop)),D_SKUs_pop[f"{columnIndex}_CUM"])
    plt.title(f"{columnIndex} Pareto curve")
    plt.xlabel("N. of SKUs")
    plt.ylabel("Popularity percentage")
    
    #save the Pareto figure
    output_figures[f"{columnIndex}_pareto"] = fig1
    
    
    fig2 = plt.figure()
    plt.hist(D_SKUs_pop[columnIndex])
    plt.title(f"{columnIndex} histogram")
    plt.xlabel(f"{columnIndex}")
    plt.ylabel("Frequency")
    
    #save the Pareto figure
    output_figures[f"{columnIndex}_hist"] = fig2
    
    
    return output_figures
    
            