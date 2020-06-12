# -*- coding: utf-8 -*-


root_folder="C:\\Users\\aletu\\Documents\\GitHub\\logproj"
#%% import packages from other folders
import sys
sys.path.append(root_folder)

import pandas as pd


# %% generate warehouse data
from logproj.data_generator_warehouse import generateWarehouseData
D_locations, D_SKUs, D_movements = generateWarehouseData()


# %% create folder hierarchy

#import utilities
from logproj.utilities import creaCartella
string_casestudy='TOY_DATA'
pathResults = 'C:\\Users\\aletu\\desktop'
_, root_path = creaCartella(pathResults,f"{string_casestudy}_results")
_, path_results = creaCartella(root_path,f"P8_warehouseAssessment")

# %% SET COLUMNS
timecolumn='TIMESTAMP_IN'
inout_column = 'INOUT'
x_col = 'LOCCODEX'
y_col = 'LOCCODEY'
z_col = 'LOCCODEZ'
sampling_interval = 'year'

# %% convert to datetime
import logproj.stat_time_series as ts
D_movements['PERIOD'] = pd.to_datetime(D_movements[timecolumn])
D_movements['PERIOD'] = ts.sampleTimeSeries(D_movements['PERIOD'],sampleInterval=sampling_interval)




# %% assess productivity
from logproj.P8_performanceAssessment.wh_productivity_assessment import spaceProductivity

for variableToPlot in ['popularity','QUANTITY','VOLUME','WEIGHT']:
    _, path_current = creaCartella(path_results,f"{variableToPlot}_productivity")
    
    fig_out_2D = spaceProductivity(D_movements,variableToPlot,inout_column, x_col,  y_col, z_col, graphType='2D',cleanData = False)
    fig_out_3D = spaceProductivity(D_movements,variableToPlot,inout_column, x_col,  y_col, z_col, graphType='3D',cleanData = False)
    
    
    #save figure
    for key in fig_out_2D.keys():
        fig_out_2D[key].savefig(path_current+f"\\{key}.png")  
    for key in fig_out_3D.keys():
        fig_out_3D[key].savefig(path_current+f"\\{key}.png")                                                 

# %% 1D (trend) productivity plot
from logproj.P8_performanceAssessment.wh_productivity_assessment import timeProductivity

for variableToPlot in ['popularity','QUANTITY','VOLUME','WEIGHT']:
    _, path_current = creaCartella(path_results,f"{variableToPlot}_productivity")

    fig_out_trend = timeProductivity(D_movements, variableToPlot, inout_column)
    
    #save figure
    for key in fig_out_trend.keys():
        fig_out_trend[key].savefig(path_current+f"\\{key}.png")  


            
           


