# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from logproj.DIST.globalBookingAnalysis import getCoverageStats

# %%
def itemSharePieGraph(D_mov, itemfield,capacityField ='QUANTITY'):
    
    #itemfield rappresenta i codici di famiglie di prodotto per rappresentarne la quota di mercato
    #capacityField e' un campo di capacita' per calcolare le coperture
    
    #calcolo le coperture
    accuracy, _ = getCoverageStats(D_mov,itemfield,capacityField='QUANTITY')
    
    #TEU-FEU share
    D_movType=D_mov.groupby([itemfield]).size().reset_index()
    D_movType=D_movType.rename(columns={0:'Percentage'})
    labels=D_movType[itemfield]
    sizes=D_movType.Percentage
    explode = 0.1*np.ones(len(sizes))
    fig1, ax1 = plt.subplots(figsize=(20,10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    #creo tabella per tipo di container
    D_movCode=D_mov.groupby([itemfield]).size().reset_index()
    D_movCode=D_movCode.rename(columns={0:'Quantity'})
    D_movCode=D_movCode.sort_values(['Quantity'],ascending=False)
    #D_movCode.to_excel(dirResults+'\\02-ContainerTypeStats.xlsx')
    
    D_movCode['accuracy']=[accuracy for i in range(0,len(D_movCode))]
    
    return fig1, D_movCode