# -*- coding: utf-8 -*-
import numpy as np



def AllocateSKUs(D_mov, D_part):
    '''
    allocate EQS, EQT and OPT allocation for all the SKUs

    Parameters
    ----------
    D_mov : TYPE pandas dataframe
        DESCRIPTION. movements dataframe containing ITEMCODE and QUANTITY columns
    D_part : TYPE pandas dataframe
        DESCRIPTION. sku master file containing ITEMCODE and VOLUME (volume cannot be null or zero)

    Returns
    -------
    D_mov_qty : TYPE pandas dataframe
        DESCRIPTION. dataframe containing the SKU master file with EQS, EQT and OPT values

    '''
    
    D_mov_qty = D_mov.groupby(['ITEMCODE']).sum()['QUANTITY'].to_frame().reset_index()
    D_mov_qty.columns = ['ITEMCODE','QUANTITY']
    D_mov_qty = D_mov_qty.merge(D_part, on='ITEMCODE', how='left')
    D_mov_qty['fi'] = D_mov_qty['QUANTITY']*D_mov_qty['VOLUME']
    D_mov_qty = D_mov_qty.dropna()
    D_mov_qty = D_mov_qty[D_mov_qty['fi']>0]
    
    if len(D_mov_qty)>0:
        D_mov_qty['EQS'] = 1/len(D_mov_qty)
        D_mov_qty['EQT'] = D_mov_qty['fi']/sum(D_mov_qty['fi'])
        D_mov_qty['OPT'] = np.sqrt(D_mov_qty['fi'])/sum(np.sqrt(D_mov_qty['fi']))
    return D_mov_qty



# %%

def discreteAllocationParts(D_parts, availableSpacedm3, method='OPT'):
    '''
    

    Parameters
    ----------
    D_parts : TYPE pandas dataframe
        DESCRIPTION. dataframe of the SKUs master file containing VOLUME, EQS, EQT and OPT columns
    availableSpacedm3 : TYPE float
        DESCRIPTION. available space in the same unit of measure of the parts VOLUME
    method : TYPE, optional allocation method (EQS, EQT, OPT)
        DESCRIPTION. The default is 'OPT'.

    Returns
    -------
    D_parts : TYPE pandas dataframe
        DESCRIPTION. SKUs master file with the allocated number of SKUs

    '''
    #Check the input columns of the dataframe
    checkColumns = ['VOLUME', 'EQS', 'EQT','OPT']
    for col in checkColumns:
        if col not in D_parts.columns:
            print(f"Column {col} not in dataframe D_parts")
            return []
        
    #Check the input method
    checkMethods = ['EQS', 'EQT','OPT']
    if method not in checkMethods:
        print(f"Unknown allocation method, choose between EQS, EQT, OPT")
        return []
    
    # allocate the parts
    D_parts['ALLOCATED_VOLUME'] = D_parts[method]*availableSpacedm3
    D_parts[f"N_PARTS_{method}"] = np.round(D_parts['ALLOCATED_VOLUME']/D_parts['VOLUME'],0)
    return D_parts


# %% DEVUG AREA




'''
import matplotlib.pyplot as plt


D_parts = D_parts.sort_values(by='OPT', ascending=False)
availableSpacedm3 = 1000*1000
methods = ['EQS','EQT','OPT']
for method in methods:
    D_parts = discreteAllocationParts(D_parts, availableSpacedm3, method)
    x = np.arange(0,len(D_parts))
    plt.scatter(x,D_parts[f"N_PARTS_{method}"])
    
plt.legend(methods)

'''




'''
# %% plot of EQS, EQT and OPT values
import matplotlib.pyplot as plt
D_mov_qty = D_mov_qty.sort_values(by='fi',ascending=False)
D_mov_plot = D_mov_qty.iloc[0:100]
x = np.arange(0,len(D_mov_plot))
plt.scatter(x,D_mov_plot['EQS'])
plt.scatter(x,D_mov_plot['EQT'])
plt.scatter(x,D_mov_plot['OPT'])
plt.legend(['EQS','EQT','OPT'])

'''