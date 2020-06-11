# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from logproj.ml_dataCleaning import cleanUsingIQR

# %% 3D warehouse productivity plot
def spaceProductivity(D_movements,inout_column, x_col,  y_col, z_col, graphType='2D',cleanData = False):
    '''
    

    Parameters
    ----------
    D_movements : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe with movements
    inout_column : TYPE string
        DESCRIPTION. string of the column with inout
    x_col : TYPE string
        DESCRIPTION. string of the column with x coordinates
    y_col : TYPE string
        DESCRIPTION. string of the column with y coordinates
    z_col : TYPE string
        DESCRIPTION. string of the column with z coordinates
    graphType : TYPE string, optional
        DESCRIPTION. The default is '2D'. 2D or 3D depending on the graph type
    cleanData : TYPE boolean, optional
        DESCRIPTION. The default is False. if True, IQR is used to clean popularity of each location

    Returns
    -------
    figure_output : TYPE dict
        DESCRIPTION. dictionary of output figures

    '''
                      
    figure_output={}
    #group data
    if graphType=='3D':
        D_mov = D_movements.groupby(['PERIOD',inout_column,x_col,y_col,z_col]).size().reset_index()
        D_mov.columns=['PERIOD','INOUT','LOCCODEX','LOCCODEY','LOCCODEZ','POPULARITY']
    elif graphType=='2D':
        D_mov = D_movements.groupby(['PERIOD',inout_column,x_col,y_col,]).size().reset_index()
        D_mov.columns=['PERIOD','INOUT','LOCCODEX','LOCCODEY','POPULARITY']
        
        
    
    # split data into inbound and outbound
    D_loc_positive=D_mov[D_mov[inout_column]=='+']
    D_loc_negative=D_mov[D_mov[inout_column]=='-']
    
    
    
    #render inbound figure
    if len(D_loc_positive)>0:
        
        #clean data
        if cleanData:
            D_warehouse_grouped, _ = cleanUsingIQR(D_loc_positive, features = ['POPULARITY'],capacityField=[])
        else:
            D_warehouse_grouped = D_loc_positive
        #create figures
        for period in set(D_warehouse_grouped['PERIOD']):
            #period = list(set(D_warehouse_grouped['PERIOD']))[0]
            D_warehouse_grouped_filtered = D_warehouse_grouped[D_warehouse_grouped['PERIOD']==period]
            D_warehouse_grouped_filtered['SIZE'] = 1 +(D_warehouse_grouped_filtered['POPULARITY'] - min(D_warehouse_grouped_filtered['POPULARITY']))/(0.1 + max(D_warehouse_grouped_filtered['POPULARITY'])-min( D_warehouse_grouped_filtered['POPULARITY']))
            
            #scale size
            D_warehouse_grouped_filtered['SIZE'] =3*D_warehouse_grouped_filtered['SIZE'] 
            
            #graphType 2-Dimensional
            if graphType == '2D':
                fig1 = plt.figure()
                plt.scatter(D_warehouse_grouped_filtered['LOCCODEX'],
                                   D_warehouse_grouped_filtered['LOCCODEY'],
                                   D_warehouse_grouped_filtered['SIZE'])
                plt.title(f"Warehouse INBOUND productivity, period:{period}")
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                figure_output[f"IN_productivity_2D_{period}"] = fig1
            
            #graphtype 3-Dimensional
            elif graphType == '3D':
                fig1 = plt.figure()
                fig1.add_subplot(111, projection='3d')
                plt.scatter(x = D_warehouse_grouped_filtered['LOCCODEX'],
                            y = D_warehouse_grouped_filtered['LOCCODEY'],
                            zs = D_warehouse_grouped_filtered['LOCCODEZ'],
                            s = D_warehouse_grouped_filtered['SIZE']
                                   )
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                plt.title(f"Warehouse INBOUND productivity, period:{period}")
                figure_output[f"IN_productivity_3D_{period}"] = fig1
                
    #render outbound figure
    if len(D_loc_negative)>0:
        
        #clean data
        if cleanData:
            D_warehouse_grouped, _ = cleanUsingIQR(D_loc_negative, features = ['POPULARITY'],capacityField=[])
        else:
            D_warehouse_grouped = D_loc_negative
        #create figures
        for period in set(D_warehouse_grouped['PERIOD']):
            #period = list(set(D_warehouse_grouped['PERIOD']))[0]
            D_warehouse_grouped_filtered = D_warehouse_grouped[D_warehouse_grouped['PERIOD']==period]
            D_warehouse_grouped_filtered['SIZE'] = 1 +(D_warehouse_grouped_filtered['POPULARITY'] - min(D_warehouse_grouped_filtered['POPULARITY']))/(0.1 + max(D_warehouse_grouped_filtered['POPULARITY'])-min( D_warehouse_grouped_filtered['POPULARITY']))
            
            #scale size
            D_warehouse_grouped_filtered['SIZE'] =3*D_warehouse_grouped_filtered['SIZE'] 
            
            #graphType 2-Dimensional
            if graphType == '2D':
                fig1 = plt.figure()
                plt.scatter(D_warehouse_grouped_filtered['LOCCODEX'],
                                   D_warehouse_grouped_filtered['LOCCODEY'],
                                   D_warehouse_grouped_filtered['SIZE'])
                plt.title(f"Warehouse OUTBOUND productivity, period:{period}")
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                figure_output[f"OUT_productivity_2D_{period}"] = fig1
            
            #graphtype 3-Dimensional
            elif graphType == '3D':
                fig1 = plt.figure()
                fig1.add_subplot(111, projection='3d')
                plt.scatter(x = D_warehouse_grouped_filtered['LOCCODEX'],
                            y = D_warehouse_grouped_filtered['LOCCODEY'],
                            zs = D_warehouse_grouped_filtered['LOCCODEZ'],
                            s = D_warehouse_grouped_filtered['SIZE']
                                   )
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                plt.title(f"Warehouse OUTBOUND productivity, period:{period}")
                figure_output[f"OUT_productivity_3D_{period}"] = fig1
    return figure_output

