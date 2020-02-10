# -*- coding: utf-8 -*-
#analytics for demand classification

#%%
import numpy as np
import matplotlib.pyplot as plt


#%% demand classification
def demandPatternADICV2(df_results, setTitle, draw=False):
    # df_results is a dataframe with columns:
    # - ADI (with the ADI value)
    # - CV2 (with the CV2 value)
    # - frequency (with the number of lines for each itemcode)
    
    #setTitle is a string containing the name of the dataset to generate the title of the figure
    
    fig=fig1=[]
    #calcolo risultati numerici
    numLumpy=len(df_results[(df_results.ADI<=1.32) & (df_results.CV2>0.49) ])
    numErratic=len(df_results[(df_results.ADI>1.32) & (df_results.CV2>0.49) ])
    numIntermittent=len(df_results[(df_results.ADI<=1.32) & (df_results.CV2<=0.49) ])
    numStable=len(df_results[(df_results.ADI>1.32) & (df_results.CV2<=0.49) ])
    totParts=numLumpy+numErratic+numIntermittent+numStable
    
    if draw:
        if totParts==len(df_results):
            A=np.array([[numLumpy, numErratic], [numIntermittent, numStable]])
            A_text=np.array([[f"Lumpy \n {numLumpy} parts \n Perc: {np.round(numLumpy*100/totParts, 2)} %", f"Erratic \n {numErratic} parts \n Perc: {np.round(numErratic*100/totParts, 2)} %"],
                              [f"Intermittent \n {numIntermittent} parts \n Perc: {np.round(numIntermittent*100/totParts, 2)} %", f"Stable \n {numStable} parts \n Perc: {np.round(numStable*100/totParts, 2)} %"]])
            fig, ax = plt.subplots()
            im = ax.imshow(A,cmap="YlOrRd")
            
            plt.title(f"Parts set: {setTitle}")
            
            
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            
            for i in range(0,2):
                for j in range(0,2):
                    ax.text(j, i, A_text[i, j],
                                   ha="center", va="center", color="k")
            
            
            #grafica risultati ADI e CV2
            #normalizzo i valori per dare dimensione alle bolle
            fig1 = plt.figure()
            plt.scatter(df_results['ADI'], df_results['CV2'],df_results['frequency'],color='skyblue',marker='o')
            plt.axvline(x=1.32, c='orange',linestyle='--')
            plt.axhline(y=0.49,c='orange',linestyle='--')
            plt.xlabel('ADI')
            plt.ylabel('CV2')
            plt.title(f"Demand pattern: {setTitle}")
            
        
    return fig, fig1, numLumpy, numIntermittent, numErratic, numStable