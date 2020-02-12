# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# %% PLOT TRENDS
    
def plotquantitytrend(D_time,  date_field, filterVariable, filterValue,  quantityVariable = 'sum_QUANTITY', countVariable = 'count_TIMESTAMP_IN'):
    D_temp=D_time[D_time[filterVariable]==filterValue]
    D_temp=D_temp.sort_values(date_field)
    D_temp=D_temp.reset_index(drop=True)
    D_temp=D_temp.dropna(subset=[date_field,quantityVariable])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f"ServiceType: {filterValue}")
    
    
    
    #plot empirical
    axs[0].plot(D_temp[date_field],D_temp[quantityVariable])
    axs[0].set_title('Quantity trend')
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(45)
    
    axs[1].plot(D_temp[date_field],D_temp[countVariable])
    axs[1].set_title('Lines trend')
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(45)
    
   
    
    plt.close('all')
    return fig