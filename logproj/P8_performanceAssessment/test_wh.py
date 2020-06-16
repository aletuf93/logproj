# -*- coding: utf-8 -*-


root_folder="C:\\Users\\aletu\\Documents\\GitHub\\logproj"
#%% import packages from other folders
import sys
sys.path.append(root_folder)

import pandas as pd


# %% generate warehouse data
from logproj.data_generator_warehouse import generateWarehouseData
D_locations, D_SKUs, D_movements, D_inventory = generateWarehouseData()


# %% create folder hierarchy

#import utilities
from logproj.utilities import creaCartella
string_casestudy='TOY_DATA'
pathResults = 'C:\\Users\\aletu\\desktop'
_, root_path = creaCartella(pathResults,f"{string_casestudy}_results")
_, path_results = creaCartella(root_path,f"P8_warehouseAssessment")

# %% SET COLUMNS MOVEMENTS
timecolumn_mov='TIMESTAMP_IN'
itemcodeColumns_mov='ITEMCODE'
inout_column_mov = 'INOUT'
x_col_mov = 'LOCCODEX'
y_col_mov = 'LOCCODEY'
z_col_mov = 'LOCCODEZ'
sampling_interval = 'year'

# %% SET COLUMNS SKUS
itemcodeColumns_sku='ITEMCODE'

# %% SET COLUMNS INVENTORY
itemcodeColumns_inv = 'ITEMCODE'

# %% convert to datetime
import logproj.stat_time_series as ts
D_movements['PERIOD'] = pd.to_datetime(D_movements[timecolumn_mov])
D_movements['PERIOD'] = ts.sampleTimeSeries(D_movements['PERIOD'],sampleInterval=sampling_interval)




# %% assess productivity
from logproj.P8_performanceAssessment.wh_productivity_assessment import spaceProductivity

for variableToPlot in ['popularity','QUANTITY','VOLUME','WEIGHT']:
    _, path_current = creaCartella(path_results,f"{variableToPlot}_productivity")
    
    fig_out_2D = spaceProductivity(D_movements,variableToPlot,inout_column_mov, x_col_mov,  y_col_mov, z_col_mov, graphType='2D',cleanData = False)
    fig_out_3D = spaceProductivity(D_movements,variableToPlot,inout_column_mov, x_col_mov,  y_col_mov, z_col_mov, graphType='3D',cleanData = False)
    
    
    #save figure
    for key in fig_out_2D.keys():
        fig_out_2D[key].savefig(path_current+f"\\{key}.png")  
    for key in fig_out_3D.keys():
        fig_out_3D[key].savefig(path_current+f"\\{key}.png")                                                 

# %% 1D (trend) productivity plot
from logproj.P8_performanceAssessment.wh_productivity_assessment import timeProductivity

for variableToPlot in ['popularity','QUANTITY','VOLUME','WEIGHT']:
    _, path_current = creaCartella(path_results,f"{variableToPlot}_productivity")

    fig_out_trend = timeProductivity(D_movements, variableToPlot, inout_column_mov)
    
    #save figure
    for key in fig_out_trend.keys():
        fig_out_trend[key].savefig(path_current+f"\\{key}.png")  
        
        
        
        
# %% generate inventory curve
from logproj.P8_performanceAssessment.wh_inventory_assessment import updatePartInventory
D_SKUs= updatePartInventory(D_SKUs,D_movements,D_inventory,timecolumn_mov,itemcodeColumns_sku,itemcodeColumns_mov,itemcodeColumns_inv) 
        
# %% POPULARITY INDEX

_, path_current = creaCartella(path_results,f"SKUs indices")

from logproj.P8_performanceAssessment.wh_inventory_assessment import updatePopularity, whIndexParetoPlot


D_SKUs = updatePopularity(D_SKUs)

#POPULARITY IN
output_figures = whIndexParetoPlot(D_SKUs,'POP_IN')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png")  
        
#POPULARITY IN
output_figures = whIndexParetoPlot(D_SKUs,'POP_OUT')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png")  

# %% ORDER COMPLETION INDEX
from logproj.P8_performanceAssessment.wh_inventory_assessment import updateOrderCompletion
D_SKUs = updateOrderCompletion(D_SKUs, D_movements)

#OC
output_figures = whIndexParetoPlot(D_SKUs,'OC')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 



# %%
import numpy as np
import matplotlib.pyplot as plt


    
    
    
    
# %%
for i in range(0,len(D_SKUs)):
    #i=33159
    print(i)

    part = D_SKUs.iloc[i]['ITEMCODE']
    if isinstance(part,np.int64): #mongodb non puo' gestire gli int64
        part=int(part)



    #calculate the order completion (OC) index
    OC = calculateOrderCompletion(D_mov, part, itemfield='ITEMCODE', ordercodefield='ORDERCODE')


    #retrieve the inventory
    I_t = D_SKUs.iloc[i]['INVENTORY_QUANTITY']

    #proseguo solo se ho una lista e non un nullo a database
    if isinstance(I_t,list):
        #bootstrap the inventory curve
        min_real_I_t, max_real_I_t, avg_real_I_t, std_real_I_t, min_probabilistic_I_t, max_probabilistic_I_t, avg_probabilistic_I_t, std_probabilistic_I_t = returnProbabilisticInventory(I_t)

        #calculate the interarrival time (average covering)
        mean_interarrival_time_in, std_interarrival_time_in, _ = assessInterarrivalTime(I_t)
        
        #calculate the fourier analysis of the inventory series
        carrier, period = fourierAnalysisInventory(I_t)

        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            #POP_IN, POP_OUT, POP_IN_TOT, POP_OUT_TOT = calculatePopularity(movements['QUANTITY'])

            #calculate the COI
            COI_IN, COI_OUT = calculateCOI(I_t)

            #calculate the TURN
            TURN = calculateTurn(I_t)

        #update the database
        model_log.part.objects(ITEMCODE=part).update(INVENTORY_REAL_MIN=min_real_I_t,
                                                     INVENTORY_REAL_AVG=avg_real_I_t,
                                                     INVENTORY_REAL_MAX=max_real_I_t,
                                                     INVENTORY_REAL_STD = std_real_I_t,
                                                     INVENTORY_PROB_MIN=min_probabilistic_I_t,
                                                     INVENTORY_PROB_AVG=avg_probabilistic_I_t,
                                                     INVENTORY_PROB_MAX=max_probabilistic_I_t,
                                                     INVENTORY_PROB_STD = std_probabilistic_I_t,
                                                     INVENTORY_COVERING_AVG = mean_interarrival_time_in,
                                                     INVENTORY_COVERING_STD = std_interarrival_time_in,
                                                     POP_IN=POP_IN,
                                                     POP_OUT=POP_OUT,
                                                     COI_IN=COI_IN,
                                                     COI_OUT=COI_OUT,
                                                     TURN=TURN,
                                                     OC = OC,
                                                     POP_IN_TOT = POP_IN_TOT,
                                                     POP_OUT_TOT = POP_OUT_TOT,
                                                     FOURIER_CARRIER =carrier,
                                                     FOURIER_PERIOD = period
                                                     )


            
           


