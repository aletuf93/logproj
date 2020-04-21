# -*- coding: utf-8 -*-

# %%
#specify root folder path
#root_folder="D:\\OneDrive - Alma Mater Studiorum Università di Bologna\\ACADEMICS\\[514]Dottorato\\Projects\\Z_WAREHOUSE\\00_SOFTWARE\\GitHub\\ZENON"
root_folder="C:\\Users\\aletu\\Documents\\GitHub\\OTHER\\ZENON"

#%% import packages from other folders
import sys
sys.path.append(root_folder)


#%% import packages from other folders

import pandas as pd
import numpy as np

from scipy.stats import poisson

#import matplotlib.pyplot as plt
#import database.mongo_queries as qq
#import database.models.odm_logistics_mongo as model_log


import logproj.stat_time_series as ts
from logproj.P1_familyProblem.part_classification import returnsparePartclassification

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
#%% ESTIMATE INVENTORY FUNCTION

def returnInventoryPart(D_mov_part, D_inv_part, timeLineDays):
    '''
    Defines the inventory function (grouped by day) of a part

    Parameters
    ----------
    D_mov_part : TYPE: pandas dataframe
        DESCRIPTION. dataframe of movements of a single part (ITEMCODE)
    D_inv_part : TYPE: pandas dataframe
        DESCRIPTION. dataframe of inventory of a single part (ITEMCODE) already grouped by TIMESTAMP and ITEMCODE
    timeLineDays : TYPE Pandas dataframe
        DESCRIPTION. dataframe wih a column TIMELINE having an aggregation of all the days to generate the inventory array
    Returns
    -------
    array_days : TYPE list
        DESCRIPTION. list of days where the inventory is reconstructed
    array_inventory : TYPE list
        DESCRIPTION. list of inventory values where the inventory is reconstructed

    '''

    #considero solo le righe con il segno
    D_mov_part=D_mov_part[D_mov_part['INOUT'].isin(['+','-'])]
    #se ho almeno un movimento
    if len(D_mov_part)>0:
        #identifico i segni dei movimenti
        D_mov_part['MOVEMENT'] = D_mov_part['INOUT'].astype(str) + D_mov_part['QUANTITY'].astype(str)
        D_mov_part['MOVEMENT']=D_mov_part['MOVEMENT'].astype(float)

        #raggruppo su base giornaliera
        D_mov_part['PERIOD'] = ts.sampleTimeSeries(D_mov_part['TIMESTAMP_IN'],'day')
        D_mov_part_grouped = D_mov_part.groupby(['PERIOD']).sum()['MOVEMENT'].reset_index()
        D_mov_part_grouped = D_mov_part_grouped.sort_values(by='PERIOD')

        #definisco un livello di inventario positivo in base ai movimenti
        D_mov_part_grouped['INVENTORY'] = D_mov_part_grouped['MOVEMENT'] - min(D_mov_part_grouped['MOVEMENT'])

        #se non ho movimenti setto a zero
    else:
        D_mov_part_grouped = pd.DataFrame([[timeLineDays['TIMELINE'].iloc[0],0]],columns=['PERIOD','INVENTORY'])

    #faccio join con la linea temporale
    D_inventory_part = timeLineDays.merge(D_mov_part_grouped, how='left', left_on='TIMELINE', right_on='PERIOD')

    #uso forward fill per riempire i nulli: se non ho inventari calcolati, fa fede l'ultimo inventario calcolato
    D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'].fillna(method='ffill')



    #se ho almeno un punto di giacenza correggo la stima della curva di inventario
    if len(D_inv_part)>0:
        D_inv_part['PERIOD'] = ts.sampleTimeSeries(D_inv_part['TIMESTAMP'],'day')
        D_inv_part = D_inv_part.merge(D_inventory_part, how='left', left_on='PERIOD', right_on='TIMELINE')

        #per ogni punto di giacenza noto tento di sistemare la curva di inventario
        for index, row in D_inv_part.iterrows():

            #verifico se ho sottostimato il livello di inventario. Non posso aver
            # sorastimato perche' ho gia' portato tutto sopra lo zero
            if row.QUANTITY > row.INVENTORY:
                gap = row.QUANTITY - row.INVENTORY
                D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'] + gap


    array_days = list(D_inventory_part['TIMELINE'])
    array_inventory = list(D_inventory_part['INVENTORY'])

    return array_days, array_inventory






# %% SIMULATE INVENTORY FUNCTION
def returnProbabilisticInventory(I_t, iterations=30):
    '''


    Parameters
    ----------
    I_t : TYPE list
        DESCRIPTION. inventory function with an element for each day
    iterations : TYPE int
        DESCRIPTION. number of iterations to simulate the inventory function
    Returns
    -------
    min_real_I_t : TYPE int
        DESCRIPTION. min value from the real inventory function I_t
    max_real_I_t : TYPE int
        DESCRIPTION. max value from the real inventory function I_t
    avg_real_I_t : TYPE int
        DESCRIPTION. average value from the real inventory function I_t
    min_probabilistic_I_t : TYPE int
        DESCRIPTION. min value from the probabilistic inventory function
    max_probabilistic_I_t : TYPE int
        DESCRIPTION. max value from the probabilistic inventory function
    avg_probabilistic_I_t : TYPE int
        DESCRIPTION. average value from the probabilistic inventory function

    '''

    #initialise output
    #compare simulated and real values
    min_real_I_t = max_real_I_t = avg_real_I_t =std_real_I_t=0
    min_probabilistic_I_t = max_probabilistic_I_t = avg_probabilistic_I_t=std_probabilistic_I_t=0


    I_t_cleaned = [x for x in I_t if str(x) != 'nan'] #remove nan inventories (e.g. at the beginning of the series if the part is not in the WH)


    #generate the movement function
    M_t = movementfunctionfromInventory(I_t_cleaned)
    
    # plot movement and inventory functions
    #plt.plot(M_t)
    #plt.plot(I_t_cleaned)

    #define the outbound quantity distribution
    M_t_out=M_t[M_t['QUANTITY']<0]
    if len(M_t_out)>0:
        #identify quantity parameters
        mean_qty_out = float(np.abs(np.mean(M_t_out)))
        sigma_qty_out = float(np.std(M_t_out))
        if sigma_qty_out==0: #having only an observation, use as sigma half of the mean
            sigma_qty_out = mean_qty_out/2

        #identify time parameters
        ADI_OUT =len(M_t_out)/len(M_t)
        interarrival_time = []
        for j in range(1,len(M_t_out)):
            interarrival_time.append(M_t_out.index[j] - M_t_out.index[j-1])
        mean_interarrival_time_out = np.mean(interarrival_time)
        std_interarrival_time_out = np.std(interarrival_time)

        if std_interarrival_time_out==0: #having only an observation, use as sigma half of the mean
            std_interavvial_time_out=mean_interarrival_time_out/2


        #depending on the demand pattern, simulates the inventory behaviour
        ADI = ADI_OUT
        CV2 = sigma_qty_out/mean_qty_out
        demand_pattern = returnsparePartclassification(ADI=ADI, CV2=CV2)

        #generates the days of the market demand
        min_inventory=[]
        max_inventory=[]
        avg_inventory=[]
        std_inventory=[]


        for iteration in range(0,iterations):
            # if intermitten or lumpy, use a poisson to generate interarrival times
            if demand_pattern in(["INTERMITTENT","LUMPY"]):

                #integer values are allowed due to Poisson distribution for out_days
                out_days = poisson.rvs(mu=ADI_OUT, size=len(I_t))

            #if stable or erratic, use a gaussian distribution to generate interarrival times
            elif demand_pattern in(["ERRATIC","STABLE"]):

                #only 0/1 values are allowed for out_days
                out_days = np.zeros(len(I_t))
                dayPointer=0
                while dayPointer<len(out_days):
                    waitingDays= int(np.round(np.random.normal(mean_interarrival_time_out, std_interavvial_time_out),0))
                    dayPointer=dayPointer+waitingDays
                    if dayPointer<len(out_days):
                        out_days[dayPointer]=1

            else:
                print("Error, demand pattern not found")

            #generates the demand values
            out_quantities = np.round(np.random.normal(mean_qty_out, sigma_qty_out,len(out_days)),0)
            #generates probabilistic outbound movements
            M_prob = out_quantities * out_days

            #calculates the optimal inventory function
            I_prob = [j if j>0 else np.nan for j in M_prob]
            I_prob = pd.DataFrame(I_prob,columns=['INVENTORY'])
            if len(I_prob.dropna())>0:
                I_prob['INVENTORY'] = I_prob['INVENTORY'].fillna(method='bfill') #defines the values of the inventory value
                I_prob['INVENTORY'] = I_prob['INVENTORY'].fillna(method='ffill') #fill nan values after the last outbound movement
                min_inventory.append(min(I_prob[I_prob['INVENTORY']>0]['INVENTORY'])) #ignore zero inventory values
                max_inventory.append(max(I_prob['INVENTORY']))
                avg_inventory.append(np.mean( I_prob['INVENTORY']))
                std_inventory.append(np.std( I_prob['INVENTORY']))

        #compare simulated and real values
        min_real_I_t = np.nanmin([j for j in I_t if j>0])
        max_real_I_t = np.nanmax(I_t)
        avg_real_I_t = np.nanmean(I_t)
        std_real_I_t = np.nanstd(I_t)

        min_probabilistic_I_t = np.nanmin(min_inventory)
        max_probabilistic_I_t = np.nanmax(max_inventory)
        avg_probabilistic_I_t = np.nanmean(avg_inventory)
        std_probabilistic_I_t = np.nanmean(std_inventory)



    #print(f"REAL: ({min_real_I_t},{avg_real_I_t},{max_real_I_t})")
    #print(f"OPTIMAL ({min_probabilistic_I_t},{avg_probabilistic_I_t},{max_probabilistic_I_t})")
    return min_real_I_t, max_real_I_t, avg_real_I_t, std_real_I_t, min_probabilistic_I_t, max_probabilistic_I_t, avg_probabilistic_I_t, std_probabilistic_I_t
