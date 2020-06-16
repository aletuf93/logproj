# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# %%
def returnInventoryRiskFromInventoryFunction(inventory_values, inventory):
    '''
    

    Parameters
    ----------
    inventory_values : TYPE list
        DESCRIPTION. list of float with inventory values
    inventory : TYPE float
        DESCRIPTION.  value of invnetory 

    Returns
    -------
    inventory : TYPE float
        DESCRIPTION. risk value associated with the value of inventory

    '''
    inventory=float(inventory)
    #build empirical pdf and cdf
    D_inventory = pd.DataFrame(inventory_values,columns=['INVENTORY'])
    D_inventory = D_inventory.groupby('INVENTORY').size().to_frame().reset_index()
    D_inventory.columns = ['INVENTORY','FREQUENCY']
    D_inventory=D_inventory.sort_values(by=['INVENTORY'])
    D_inventory['PROB']=D_inventory['FREQUENCY']/sum(D_inventory['FREQUENCY'])
    D_inventory['CUMULATIVE']=D_inventory['PROB'].cumsum()
    
    #calculate the inventory quantity
    
    if(inventory>max(inventory_values)):
        D_opt_x_max=D_inventory.iloc[-1]
    else:
        D_opt_x_max=D_inventory[D_inventory['INVENTORY']>=inventory].iloc[0]
    
    if(inventory<min(inventory_values)):
        D_opt_x_min=D_inventory.iloc[0]
    else:
        D_opt_x_min=D_inventory[D_inventory['INVENTORY']<inventory].iloc[-1]
    
    x_array=[D_opt_x_min['INVENTORY'],D_opt_x_max['INVENTORY']]
    y_array=[D_opt_x_min['CUMULATIVE'],D_opt_x_max['CUMULATIVE']]
    
    prob=np.interp(inventory, x_array, y_array)
    risk=1-prob
    return risk






# %%
def returnInventoryValueFromInventoryFunctionRisk(inventory_values, risk):
    '''
    

    Parameters
    ----------
    inventory_values : TYPE list
        DESCRIPTION. list of float with inventory values
    risk : TYPE float
        DESCRIPTION.  value of risk 

    Returns
    -------
    inventory : TYPE float
        DESCRIPTION. inventory value associated with the risk

    '''
    #build empirical pdf and cdf
    D_inventory = pd.DataFrame(inventory_values,columns=['INVENTORY'])
    D_inventory = D_inventory.groupby('INVENTORY').size().to_frame().reset_index()
    D_inventory.columns = ['INVENTORY','FREQUENCY']
    D_inventory=D_inventory.sort_values(by=['INVENTORY'])
    D_inventory['PROB']=D_inventory['FREQUENCY']/sum(D_inventory['FREQUENCY'])
    D_inventory['CUMULATIVE']=D_inventory['PROB'].cumsum()
    
    #calculate the inventory quantity
    prob = 1-risk
    D_opt_x_max=D_inventory[D_inventory['CUMULATIVE']>=prob].iloc[0]
    D_opt_x_min=D_inventory[D_inventory['CUMULATIVE']<prob].iloc[-1]
    
    x_array=[D_opt_x_min['CUMULATIVE'],D_opt_x_max['CUMULATIVE']]
    y_array=[D_opt_x_min['INVENTORY'],D_opt_x_max['INVENTORY']]
    inventory=np.interp(prob, x_array, y_array)
    return inventory

# %%
def returnStockoutRisk(x,a,b,c):
    '''
    returns the CDF value of a triangular distribution

    Parameters
    ----------
    x : TYPE float
        DESCRIPTION. independent variable of the CDF
    a : TYPE float 
        DESCRIPTION. min value of the triangular distribution
    b : TYPE float
        DESCRIPTION. max value of the triangular sitribution
    c : TYPE float
        DESCRIPTION. mode of the triangular distribution

    Returns
    -------
    TYPE float
        DESCRIPTION. value of risk associated with the inventory level x

    '''
    probability = np.nan
    if x<=a:
        probability= 0
    elif (a<x) & (x<=c):
        probability= ((x-a)**2)/((b-a)*(c-a)) 
    elif (c<x) & (x<b):
        probability= 1 - ((b-x)**2)/((b-a)*(b-c))
    elif (x>=b):
        probability= 1
    else:
        print("Error in the CDF")
    return 1-probability

# %%
def returnStockQuantityFromRisk(risk,a,b,c):
    '''
    

    Parameters
    ----------
    risk : TYPE float
        DESCRIPTION. value of risk associated to the probability distribution (inverse of the probability)
     a : TYPE float 
        DESCRIPTION. min value of the triangular distribution
    b : TYPE float
        DESCRIPTION. max value of the triangular sitribution
    c : TYPE float
        DESCRIPTION. mode of the triangular distribution

    Returns
    -------
    x : TYPE float
        DESCRIPTION. value of inventory associated with the risk 

    '''
    

    u = 1-risk
    x = np.nan
    if (0<=u) & (u<((c-a)/(b-a))) :
        x= a + np.sqrt((b-a)*(c-a)*u)
    elif (((c-a)/(b-a))<=u) & (u<=1):
        x = b - np.sqrt((b-a)*(b-c)*(1-u))
    
    else:
        print("Error in the CDF")
    return x