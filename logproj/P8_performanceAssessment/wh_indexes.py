# -*- coding: utf-8 -*-
import numpy as np

import logproj.stat_time_series as ts
from logproj.information_framework import movementfunctionfromInventory

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

# %% fourier analysis of the inventory curve
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
