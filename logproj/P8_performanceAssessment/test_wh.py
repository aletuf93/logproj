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

        
# %% analyse the inventory behaviour
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
        
        
# %% STUDY CORRELATIONS
_, path_current = creaCartella(path_results,f"Correlations")
from logproj.P8_performanceAssessment.wh_inventory_assessment import buildLearningTablePickList
# extract learning table for each picking list
D_learning=buildLearningTablePickList(D_movements)
D_learning.to_excel(path_current+"\\learning table.xlsx")

# %% histograms
from logproj.P8_performanceAssessment.wh_inventory_assessment import histogramKeyVars
output_figures = histogramKeyVars(D_learning)
for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 


# %% correlation matrices
from logproj.P8_performanceAssessment.wh_inventory_assessment import exploreKeyVars
output_figures = exploreKeyVars(D_learning)
for key in output_figures.keys():
        output_figures[key].savefig(path_current+f"\\{key}.png") 



# %% ALLOCATION EQS, EQT, OPT
_, path_current = creaCartella(path_results,f"Allocation")

from logproj.P5_powerProblem.warehouseAllocation import AllocateSKUs

D_allocation = AllocateSKUs(D_movements, D_SKUs)
D_allocation.to_excel(path_current+"\\allocation continuous.xlsx")

# %% DISCRETE ALLOCATION
from logproj.P5_powerProblem.warehouseAllocation import discreteAllocationParts

availableSpacedm3 = 10000 #available space in dm3
for method in ['EQS', 'EQT', 'OPT']:
    D_allocated = discreteAllocationParts(D_allocation, availableSpacedm3, method='OPT')
    D_allocated.to_excel(path_current+f"\\discrete allocation {method}.xlsx")

# %% ASSIGNMENT
from logproj.P6_placementProblem.warehouse_graph_definition import calculateStorageLocationsDistance, extractIoPoints
_, path_current = creaCartella(path_results,f"Assignment")

#prepare input table locations
D_loc = D_layout.append(D_IO)
D_loc.columns=[i.upper() for i in D_loc.columns ]
input_loccodex, input_loccodey, output_loccodex, output_loccodey = extractIoPoints(D_loc)
D_loc=calculateStorageLocationsDistance(D_loc,input_loccodex, input_loccodey, output_loccodex, output_loccodey)



# %% calculate storage locations classes

from logproj.P2_assignmentProblem.warehousing_ABC_saving import defineABCclassesOfStorageLocations 
AclassPerc = 0.20
BclassPerc = 0.50
D_loc = defineABCclassesOfStorageLocations(D_loc, AclassPerc=AclassPerc, BclassPerc=BclassPerc)
D_loc.to_excel(path_current+f"\\locations assignment.xlsx")

# %% calculate part classes
from logproj.P2_assignmentProblem.warehousing_ABC_saving import defineABCclassesOfParts
columnWeightList=['POP_IN','POP_OUT']
AclassPerc = 0.20
BclassPerc = 0.50
D_assignment = defineABCclassesOfParts(D_SKUs,columnWeightList, AclassPerc = AclassPerc, BclassPerc = BclassPerc)
D_assignment.to_excel(path_current+f"\\locations assignment.xlsx")

# %% calculate saving for different Pareto cuts
from logproj.P8_performanceAssessment.wh_inventory_assessment import plotSavingABCclass
p =max(D_loc['LOCCODEX']) - min(D_loc['LOCCODEX'])
q =max(D_loc['LOCCODEY']) - min(D_loc['LOCCODEY'])

figure_output=plotSavingABCclass(p,q,D_SKUs)
for key in figure_output.keys():
        figure_output[key].savefig(path_current+f"\\{key}.png") 


















            
           


