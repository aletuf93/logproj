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

# %% COI INDEX
from logproj.P8_performanceAssessment.wh_inventory_assessment import updateCOI
D_SKUs = updateCOI(D_SKUs)

#COI IN
output_figures = whIndexParetoPlot(D_SKUs,'COI_IN')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 
        
#COI OUT
output_figures = whIndexParetoPlot(D_SKUs,'COI_OUT')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 


# %% TURN INDEX
from logproj.P8_performanceAssessment.wh_inventory_assessment import updateTURN
D_SKUs = updateTURN(D_SKUs)

#COI IN
output_figures = whIndexParetoPlot(D_SKUs,'TURN')

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 
        


# %% INVENTORY PARAMETERS

from logproj.P8_performanceAssessment.wh_inventory_assessment import updateInventoryParams
D_SKUs = updateInventoryParams(D_SKUs)


# %% INTERARRIVAL TIME

from logproj.P8_performanceAssessment.wh_inventory_assessment import updateInterarrivalTime
D_SKUs = updateInterarrivalTime(D_SKUs)

# %% INTERARRIVAL TIME

from logproj.P8_performanceAssessment.wh_inventory_assessment import updateFourieranalysis
D_SKUs = updateFourieranalysis(D_SKUs)


# %% save sku table
D_SKUs.to_excel(path_current+"\\SKUs.xlsx")





# %% update global inventory
from logproj.P8_performanceAssessment.wh_inventory_assessment import updateGlobalInventory

_, path_current = creaCartella(path_results,f"Inventory")

D_global_inventory = updateGlobalInventory(D_SKUs,inventoryColumn='INVNETORY_QTY')
D_global_inventory.to_excel(path_current+"\\global inventory.xlsx")

        
# %% analyse the inventory behavious
from logproj.P8_performanceAssessment.wh_inventory_assessment import inventoryAnalysis

output_figures = inventoryAnalysis(D_global_inventory)

for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 
        
        
# %% LAYOUT ANALYSIS
from logproj.P6_placementProblem.warehouse_graph_definition import prepareCoordinates
import numpy as np
_, path_current = creaCartella(path_results,f"Layout")


D_layout, D_IO, D_fake,  allLocs = prepareCoordinates(D_locations)
D_layout['aislecodex'] =np.nan 


# %% PREPARE DATA FOR GRAPH DEFINITION
from logproj.P6_placementProblem.warehouse_graph_definition import defineWHgraph
G, D_res, D_layout = defineWHgraph(D_layout=D_layout, 
              D_IO=D_IO, 
              D_fake=D_fake,
              allLocs = len(D_locations), 
              draw=False, 
              arcLabel=False, 
              nodeLabel=False, 
              trafficGraph=True)

# %% DEFINE THE GRAPH
from  logproj.ml_graphs import printGraph
# print the graph
distance=weight='length'
title='Warehouse graph'
printNodecoords=False

fig1 = printGraph(G, 
           distance, 
           weight, 
           title, 
           arcLabel=False, 
           nodeLabel=False, 
           trafficGraph=False,
           printNodecoords=True,
           D_layout=D_layout)
fig1.savefig(path_current+"//layout_graph.png") 

# %% DEFINE TRAFFIC CHART
fig2 = printGraph(G, 
           distance, 
           weight, 
           title, 
           arcLabel=False, 
           nodeLabel=False, 
           trafficGraph=True,
           printNodecoords=False,
           D_layout=D_layout)
fig2.savefig(path_current+"//traffic_graph.png") 

# %% CALCULATE SAVINGS FROM EXCHANGE
from logproj.P6_placementProblem.warehouse_graph_definition import calculateExchangeSaving
D_results = calculateExchangeSaving(D_movements, D_res, G, useSameLevel=False)

# %% DRAW AS IS and TO-BE BUBBLES
from logproj.P6_placementProblem.warehouse_graph_definition import returnbubbleGraphAsIsToBe
output_figures = returnbubbleGraphAsIsToBe(D_results)
for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 
        
        
# %% POP-DIST FIGURE
from logproj.P6_placementProblem.warehouse_graph_definition import asisTobeBubblePopDist
output_figures = asisTobeBubblePopDist(D_results)
for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 



















            
           


