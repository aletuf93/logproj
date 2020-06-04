# -*- coding: utf-8 -*-

#import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import osmnx as ox

import networkx as nx
#import plotly graphs



from chart.chart_3D_surface import createFigureWith3Dsurface


from logproj.P8_performanceAssessment.utilities_movements import getCoverageStats
from logproj.ml_dataCleaning import cleanUsingIQR

from sklearn.metrics import mean_squared_error

#import clustering methods
from sklearn import cluster
from sklearn.mixture import GaussianMixture

# %% MERCATOR PROJECTION
def mercatorProjection(latitude, longitude):
    
    R=6378.14 #raggio della terra
    e=0.0167 #eccentricita' della terra
    
    lon_rad=(np.pi/180)*longitude
    lat_rad=(np.pi/180)*latitude
    
    x=R*lon_rad
    y=R*np.log(((1-e*np.sin(lat_rad))/(1+e*np.sin(lat_rad)))**(e/2)*np.tan(np.pi/4 + lat_rad/2))
    return x,y




# %% DEFINE FUNCTIONS FOR OPTIMAL LOCATION
def optimalLocationRectangularDistance(D_filtered, latCol, lonCol, weightCol):
    #the function returns the optimal location based on rectangular distances
    #D_filtered is a dataframe with columns LATITUDE, LONGITUDE and FLOWS indicating the location and the intensity of the flow
    
    #optimal location
    op_w=sum(D_filtered[weightCol])/2 # identify the median of the sum of weights
    
    #identify optimal latitude
    if len(D_filtered)>1: #when there are more than a single point
        D_filtered=D_filtered.sort_values(by=latCol, ascending=True) #sort by latitude
        D_filtered['X_cumsum']=D_filtered[weightCol].cumsum() #calculate the cumulated sum
        
        #identify the LATITUDE closer to the optimal location 
        D_opt_x_max=D_filtered[D_filtered['X_cumsum']>=op_w].iloc[0]
        D_opt_x_min=D_filtered[D_filtered['X_cumsum']<op_w].iloc[-1]
        
        x_array=[D_opt_x_min['X_cumsum'],D_opt_x_max['X_cumsum']]
        y_array=[D_opt_x_min[latCol],D_opt_x_max[latCol]]
        lat_optimal=np.interp(op_w, x_array, y_array)
        
        #identify the LONGITUDE closer to the optimal location 
        D_filtered=D_filtered.sort_values(by=lonCol, ascending=True) #sort by latitude
        D_filtered['Y_cumsum']=D_filtered[weightCol].cumsum() #calculate the cumulated sum
        
        D_opt_x_max=D_filtered[D_filtered['Y_cumsum']>=op_w].iloc[0]
        D_opt_x_min=D_filtered[D_filtered['Y_cumsum']<op_w].iloc[-1]
        
        x_array=[D_opt_x_min['Y_cumsum'],D_opt_x_max['Y_cumsum']]
        y_array=[D_opt_x_min[lonCol],D_opt_x_max[lonCol]]
        lon_optimal=np.interp(op_w, x_array, y_array)
        
    else: #with a single point take the coordinates of the point
        lat_optimal = float(D_filtered.iloc[0][latCol])
        lon_optimal = float(D_filtered.iloc[0][lonCol])
        
    return lat_optimal, lon_optimal

def optimalLocationGravityProblem(D_filtered, latCol, lonCol, weightCol):
    #the dunction calculate the optimal location with squared euclidean distances 
    #D_filtered is a dataframe with flow intensity, latitude and longitude
    #latCol is a string with the name of the columns woith latitude
    #loncol is a string with the name of the columns with longitude
    #weightCol is a string with the name of the columns with the flow intensity
    #print(D_filtered)
    D_filtered_notnan=D_filtered.dropna(subset=[latCol,lonCol,weightCol])
    D_filtered_notnan = D_filtered_notnan[D_filtered_notnan[weightCol]>0]
    if len(D_filtered_notnan)>0:
        lat_optimal=sum(D_filtered_notnan[latCol]*D_filtered_notnan[weightCol])/sum(D_filtered_notnan[weightCol])
        lon_optimal=sum(D_filtered_notnan[lonCol]*D_filtered_notnan[weightCol])/sum(D_filtered_notnan[weightCol])
    else:
        lat_optimal = lon_optimal = 0
    return lat_optimal, lon_optimal
    
    
def funcGKuhn(wi,xj_1,yj_1,ai,bi):
    #implements the fungtion g in the kuhn procedure for euclidean distances
    return wi/((xj_1-ai)**2+(yj_1-bi)**2)
 
def optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, weightCol):
    
    #the dunction calculate the optimal location with euclidean distances using the kuhn procedure
    #D_filtered is a dataframe with flow intensity, latitude and longitude
    #latCol is a string with the name of the columns woith latitude
    #loncol is a string with the name of the columns with longitude
    #weightCol is a string with the name of the columns with the flow intensity
    
    #remove null values
    D_filtered_notnan=D_filtered.dropna(subset=[latCol,lonCol,weightCol])
    
    #identifico la prima soluzione del gravity problem
    lat_optimal_0, lon_optimal_0 = optimalLocationGravityProblem(D_filtered_notnan, latCol, lonCol, weightCol)
    
    xj_1=lon_optimal_0
    yj_1=lat_optimal_0
    wi = D_filtered_notnan[weightCol]
    ai=D_filtered_notnan[lonCol]
    bi=D_filtered_notnan[latCol]
    
    #iterazione procedura di kuhn per longitudine
    diff_x=1 #un grado di latitudine e' circa 111 km
    while diff_x>0.01:
        lon_optimal_j = sum(funcGKuhn(wi,xj_1,yj_1,ai,bi)*ai)/sum(funcGKuhn(wi,xj_1,yj_1,ai,bi))
        diff_x = np.abs(xj_1-lon_optimal_j)
        #print(diff_x)
        xj_1=lon_optimal_j
        
     #iterazione procedura di kuhn per latitudine
    diff_x=1 #un grado di latitudine e' circa 111 km
    while diff_x>0.01:
        lat_optimal_j = sum(funcGKuhn(wi,xj_1,yj_1,ai,bi)*bi)/sum(funcGKuhn(wi,xj_1,yj_1,ai,bi))
        diff_x = np.abs(yj_1-lat_optimal_j)
        #print(diff_x)
        yj_1=lat_optimal_j
        
    
    return lat_optimal_j, lon_optimal_j

# %% DEFINE FUNCTIONS TO CALCULATE DISTANCES  
def func_rectangularDistanceCost(x, y, x_opt, y_opt, wi):
    #return cost values with rectangular distances
    return (np.abs(x-x_opt) + np.abs(y-y_opt))*wi

def func_gravityDistanceCost(x, y, x_opt, y_opt, wi):
    # return cost values with squared euclidean distances
    return ((x-x_opt)**2 + (y-y_opt)**2)*wi

def func_euclideanDistanceCost(x, y, x_opt, y_opt, wi):
    #return cost values with euclidean distance
    return np.sqrt((x-x_opt)**2 + (y-y_opt)**2)*wi

# %% DEFINE THE DISTANCE TABLE AND THE BEST ESTIMATOR
def defineDistanceTableEstimator(D_mov,lonCol_From_mov,latCol_From_mov,lonCol_To_mov,latCol_To_mov,G,cleanOutliersCoordinates=False,capacityField='QUANTITY'):
    
    '''
    D_mov is the dataframe with movements
    lonCol_From_mov is the name of the D_mov dataframe with longitude of the loading node
    latCol_From_mov is the name of the D_mov dataframe with latitude of the loading node
    lonCol_To_mov is the name of the D_mov dataframe with longitude of the discharging node
    latCol_To_mov is the name of the D_mov dataframe with latitude of the loading node
    G is a road graph obtained with osmnx
    cleanOutliersCoordinates is true to remove outliers in latitude and longitude
    capacityField is a field of capacity to measure the coverage statistics on it
    '''
    
    #clean data and get coverages
    analysisFieldList = [lonCol_From_mov,latCol_From_mov,lonCol_To_mov,latCol_To_mov]
    coverages,_ = getCoverageStats(D_mov,analysisFieldList,capacityField=capacityField)
    D_dist = D_mov[[lonCol_From_mov,latCol_From_mov,lonCol_To_mov,latCol_To_mov]].drop_duplicates().dropna().reset_index()
    if cleanOutliersCoordinates:
        D_dist,coverages_outl=cleanUsingIQR(D_dist, [lonCol_From_mov,latCol_From_mov,lonCol_To_mov,latCol_To_mov])
        coverages = (coverages[0]*coverages_outl[0],coverages[1]*coverages_outl[1])
    
    df_coverages = pd.DataFrame(coverages)
        
        
    D_dist['REAL_DISTANCE'] = np.nan
    D_dist['MERCATOR_X_FROM'] = np.nan
    D_dist['MERCATOR_Y_FROM'] = np.nan
    D_dist['MERCATOR_X_TO'] = np.nan
    D_dist['MERCATOR_Y_TO'] = np.nan
    
    for index, row in D_dist.iterrows():
        
        #get the coordinates
        lonFrom = row[lonCol_From_mov]
        latFrom = row[latCol_From_mov]
        lonTo = row[lonCol_To_mov]
        latTo = row[latCol_To_mov]
        
        #get the closest node on the graph
        node_from = ox.get_nearest_node(G, (latFrom,lonFrom), method='euclidean')
        node_to = ox.get_nearest_node(G, (latTo,lonTo), method='euclidean')
        length = nx.shortest_path_length(G=G, source=node_from, target=node_to, weight='length')
        D_dist['REAL_DISTANCE'].loc[index]=length
        
        #convert into mercator coordinates
        x_merc_from, y_merc_from =mercatorProjection(latFrom,lonFrom)
        x_merc_to, y_merc_to =mercatorProjection(latTo,lonTo)
        
        D_dist['MERCATOR_X_FROM'].loc[index]=x_merc_from
        D_dist['MERCATOR_Y_FROM'].loc[index]=y_merc_from
        D_dist['MERCATOR_X_TO'].loc[index]=x_merc_to
        D_dist['MERCATOR_Y_TO'].loc[index]=y_merc_to
    
    
    D_dist['EUCLIDEAN_DISTANCE'] = 1000*func_euclideanDistanceCost(D_dist['MERCATOR_X_FROM'],D_dist['MERCATOR_Y_FROM'],D_dist['MERCATOR_X_TO'],D_dist['MERCATOR_Y_TO'],1)
    D_dist['RECTANGULAR_DISTANCE'] = 1000*func_rectangularDistanceCost(D_dist['MERCATOR_X_FROM'],D_dist['MERCATOR_Y_FROM'],D_dist['MERCATOR_X_TO'],D_dist['MERCATOR_Y_TO'],1)
    D_dist['GRAVITY_DISTANCE'] = 1000*func_gravityDistanceCost(D_dist['MERCATOR_X_FROM'],D_dist['MERCATOR_Y_FROM'],D_dist['MERCATOR_X_TO'],D_dist['MERCATOR_Y_TO'],1)
    
    
    error_euclidean = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['EUCLIDEAN_DISTANCE'])
    error_rectangular = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['RECTANGULAR_DISTANCE'])
    error_gravity = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['GRAVITY_DISTANCE'])
    
    print(f"MSE EUCLIDEAN: {np.round(error_euclidean,2)}")
    print(f"MSE RECTANGULAR: {np.round(error_rectangular,2)}")
    print(f"MSE GRAVITY: {np.round(error_gravity,2)}")
    return D_dist, df_coverages

# %% DEFINE FUNCTION TO CALCULATE THE OPTIMAL LOCATION
def calculateOptimalLocation(D_table,
                             timeColumns,
                             distanceType,
                             latCol,
                             lonCol,
                             codeCol_node, 
                             descrCol_node,
                             cleanOutliers=False):
    '''
    # this function import a table D_table where each row is a node of the network
    #columns "NODE_DESCRIPTION" describe the node 
    #timeColumns e' la lista delle colonne con l'orizzonte temporale che contengono i dati di flusso
    #latCol identify the latitude of the node
    #lonCol identify the longitude of the node
    #codeCol_node is a column with description of the node (the same appearing in plantListName)
    #descrCol_node is a column with description of the node
    #cleanOutliers if True use IQR to remove latitude and longitude outliers
    
    
    # it returns a dataframe D_res with the ID, LATITUDE, LONGITUDE AND YEAR
    # for each flow adding the column COST AND FLOW representing the distance 
    # travelled (COST) and the flow intensity (FLOW). The column
    # COST_NORM is a the flows scaled between 0 and 100
    
    # it returns a dataframe D_res_optimal with the loptimal latitude and longitude for each
    # time frame, and a column COST and FLOW with the total cost (distance) and flows
    '''
    # pulisco i dati e calcolo le coperture
    output_coverages={}
    
    analysisFieldList=[latCol, lonCol]
    outputCoverages, _ = getCoverageStats(D_table,analysisFieldList,capacityField=timeColumns[0])
    D_table=D_table.dropna(subset=[latCol,lonCol])
    if cleanOutliers:
        D_table, coverages, =cleanUsingIQR(D_table, [latCol,lonCol])
        outputCoverages = (coverages[0]*outputCoverages[0],coverages[1]*outputCoverages[1])
    output_coverages['coverages'] = pd.DataFrame(outputCoverages)
    
    #sostituisco i nulli rimasti con zeri
    D_table=D_table.fillna(0)
    
    
    #identifico gli anni nella colonna dizionario
    yearsColumns = timeColumns
    
    
    # identifico le colonne utili
    D_res=pd.DataFrame(columns=[codeCol_node, descrCol_node,latCol,lonCol,'YEAR','COST',])
    D_res_optimal=pd.DataFrame(columns=['PERIOD',latCol,lonCol,'YEAR','COST','FLOW'])
    
    for year in yearsColumns:
        #year = yearsColumns[0]
        D_filter_columns=[codeCol_node,descrCol_node,latCol,lonCol,year]
        D_filtered = D_table[D_filter_columns]
        D_filtered = D_filtered.rename(columns={year:'FLOW'})
        D_filtered['YEAR']=year
        
        # define optimal location
        if distanceType.lower()=='rectangular':
            lat_optimal, lon_optimal = optimalLocationRectangularDistance(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST']=func_rectangularDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        elif distanceType.lower()=='gravity':
            lat_optimal, lon_optimal = optimalLocationGravityProblem(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST']=func_gravityDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        elif distanceType.lower()=='euclidean':
            lat_optimal, lon_optimal = optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST']=func_euclideanDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        D_res=D_res.append(D_filtered)
        
        
        D_res_optimal=D_res_optimal.append(pd.DataFrame([[f"OPTIMAL LOCATION YEAR: {year}",
                                                          lat_optimal, 
                                                          lon_optimal, 
                                                          year,
                                                          sum(D_res['COST']),
                                                          sum(D_res['FLOW']),
                                                          ]], columns=D_res_optimal.columns))
    
        
    #D_res['COST_norm']=(D_res['COST']-min(D_res['COST']))/(max(D_res['COST'])-min(D_res['COST']))*10
    D_res['FLOW_norm']=(D_res['FLOW']-min(D_res['FLOW']))/(max(D_res['FLOW'])-min(D_res['FLOW']))*100
    

    D_res=D_res.rename(columns={'COST':'COST_TOBE'})
    
    return D_res, D_res_optimal, output_coverages

# %% DEFINE FUNCTION TO CALCULATE MULTIPLE OPTIMAL LOCATIONS
def calculateMultipleOptimalLocation(D_table,
                             timeColumns,
                             distanceType,
                             latCol,
                             lonCol,
                             codeCol_node, 
                             descrCol_node,
                             cleanOutliers=False,
                             k=1,
                             method='kmeans'):
    '''
    #this function defines k facility location using an aggregation method
    
    # this function import a table D_table where each row is a node of the network
    #columns "NODE_DESCRIPTION" describe the node 
    #timeColumns e' la lista delle colonne con l'orizzonte temporale che contengono i dati di flusso
    #latCol identify the latitude of the node
    #lonCol identify the longitude of the node
    #codeCol_node is a column with description of the node (the same appearing in plantListName)
    #descrCol_node is a column with description of the node
    #cleanOutliers if True use IQR to remove latitude and longitude outliers
    # k is the number of optimal point to define
    # method is the method to cluster the points: kmeans, gmm
    
    
    # it returns a dataframe D_res with the ID, LATITUDE, LONGITUDE AND YEAR
    # for each flow adding the column COST AND FLOW representing the distance 
    # travelled (COST) and the flow intensity (FLOW). The column
    # COST_NORM is a the flows scaled between 0 and 100
    
    # it returns a dataframe D_res_optimal with the loptimal latitude and longitude for each
    # time frame, and a column COST and FLOW with the total cost (distance) and flows
    
    '''
    # pulisco i dati e calcolo le coperture
    output_coverages={}
    
    analysisFieldList=[latCol, lonCol]
    outputCoverages, _ = getCoverageStats(D_table,analysisFieldList,capacityField=timeColumns[0])
    D_table=D_table.dropna(subset=[latCol,lonCol])
    if cleanOutliers:
        D_table, coverages, =cleanUsingIQR(D_table, [latCol,lonCol])
        outputCoverages = (coverages[0]*outputCoverages[0],coverages[1]*outputCoverages[1])
    output_coverages['coverages'] = pd.DataFrame(outputCoverages)
    
    #sostituisco i nulli rimasti con zeri
    D_table=D_table.fillna(0)
    
    
    #identifico gli anni nella colonna dizionario
    yearsColumns = timeColumns
    
    #clusterizzo i punti
    
    
    if method == 'kmeans':
        km = cluster.KMeans(n_clusters=k).fit(D_table[[latCol,lonCol]])
        D_table['CLUSTER'] = pd.DataFrame(km.labels_)
        
    elif method == 'gmm':
        gmm = GaussianMixture(n_components=k, covariance_type='full').fit(D_table[[latCol,lonCol]])
        D_table['CLUSTER']=pd.DataFrame(gmm.predict(D_table[[latCol,lonCol]]))
    else:
        print("No valid clustering method")
        return [], [], []
    
    
    # identifico le colonne utili
    D_res=pd.DataFrame(columns=[codeCol_node, descrCol_node,latCol,lonCol,'YEAR','COST','CLUSTER'])
    D_res_optimal=pd.DataFrame(columns=['PERIOD',latCol,lonCol,'YEAR','COST','FLOW','CLUSTER'])
    
    #analizzo ogni cluster separatamente
    for cluster_id in set(D_table['CLUSTER']):
        #cluster_id=0
        D_table_filtered=D_table[D_table['CLUSTER']==cluster_id]
        for year in yearsColumns:
            #year = yearsColumns[0]
            D_filter_columns=[codeCol_node,descrCol_node,latCol,lonCol,year,'CLUSTER']
            D_filtered = D_table_filtered[D_filter_columns]
            D_filtered = D_filtered.rename(columns={year:'FLOW'})
            D_filtered['YEAR']=year
            
            # define optimal location
            if distanceType.lower()=='rectangular':
                lat_optimal, lon_optimal = optimalLocationRectangularDistance(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST']=func_rectangularDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            elif distanceType.lower()=='gravity':
                lat_optimal, lon_optimal = optimalLocationGravityProblem(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST']=func_gravityDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            elif distanceType.lower()=='euclidean':
                lat_optimal, lon_optimal = optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST']=func_euclideanDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            D_res=D_res.append(D_filtered)
            
            
            D_res_optimal=D_res_optimal.append(pd.DataFrame([[f"OPTIMAL LOCATION YEAR: {year}",
                                                              lat_optimal, 
                                                              lon_optimal, 
                                                              year,
                                                              sum(D_res['COST']),
                                                              sum(D_res['FLOW']),
                                                              cluster_id
                                                              ]], columns=D_res_optimal.columns))
    
        
    #D_res['COST_norm']=(D_res['COST']-min(D_res['COST']))/(max(D_res['COST'])-min(D_res['COST']))*10
    D_res['FLOW_norm']=(D_res['FLOW']-min(D_res['FLOW']))/(max(D_res['FLOW'])-min(D_res['FLOW']))*100
    

    D_res=D_res.rename(columns={'COST':'COST_TOBE'})
    
    return D_res, D_res_optimal, output_coverages
# %% COST OF AN AS IS ASSIGNMENT SCENARIO
def calculateCostASIS(D_plant,
                      latCol_plant,
                      lonCol_plant,
                      plantListName,
                      D_node,
                      nodeCol_node,
                      latCol_node,
                      lonCol_node,
                      distanceType):
    
    '''
    define the cost as-is of a network, given a cost estimator of the distance function distanceType
    a dataframe D_plant with one row for each plant of the network to evaluate.
    
    INPUT
    
    D_node = input table with one row for each node of the network. Can be a D_res output of the function calculateOptimalLocation
    latCol_node = string with the column of D_node with latitudes
    lonCol_node = string with the column of D_node with longitudes
    
    plantLatitude= string with the column of D_plant with latitudes
    plantLongitude= string with the column of D_plant with longitudes
    plantListName = string of the column of D_plant containing a list of all the ids of the nodes served by the plant
    
    distanceType='euclidean','gravity','rectangular'
    '''
       
    
    
    D_node['COST_ASIS']=np.nan
    D_node=D_node.reset_index(drop=True)
    
    
    
    #assegno ogni punto al plant che lo serve
    for plant in D_plant['_id']:
        #plant =652
        
        plant_client_list = D_plant[D_plant['_id']==plant][plantListName].iloc[0]
        plant_client_list = [str(i) for i in plant_client_list]
            
        #considero latitudine e longitude del plant
        lat_plant= D_plant[D_plant['_id']==plant][latCol_plant].iloc[0]
        lon_plant= D_plant[D_plant['_id']==plant][lonCol_plant].iloc[0]
        
        #creo una nuova colonna per identificare se servito da un determinato plant
        D_node[plant]= [str(id_nodo) in plant_client_list for id_nodo in D_node[nodeCol_node]]
        
        #D_res['all']=D_res[652].astype(int) + D_res[2615].astype(int) +D_res[603].astype(int) + D_res[610].astype(int)
        D_filtered = D_node[D_node[plant]==True]
        idx_to_upload = D_filtered.index.values
        
        
        #identifico la funzione di distanza
        if distanceType.lower()=='rectangular': func = func_rectangularDistanceCost
        elif distanceType.lower()=='gravity': func = func_gravityDistanceCost
        elif distanceType.lower()=='euclidean': func = func_euclideanDistanceCost
        
        
        distancecost = list(func(D_filtered[lonCol_node], 
                            D_filtered[latCol_node], 
                            lon_plant, 
                            lat_plant, 
                            D_filtered['FLOW'])
                            )
        for i in range(0,len(distancecost)):
            D_node['COST_ASIS'].loc[idx_to_upload[i]] = distancecost[i]
    return D_node

# %% RAPPRESENTA LINEE ISOCOSTO

def tracciaCurveIsocosto(D_res, D_res_optimal, latCol,  lonCol, distanceType, 
                         D_plant=[], plantLongitude=[], plantLatitude=[],
                         roadGraph=[]):
    
    outputFigure = {}
    X_list=[]
    Y_list=[]
    grid_list=[]
    time_list=[]
    
    year_list=list(set(D_res['YEAR']))
    year_list.sort()
    for year in year_list:
        #year = list(set(D_res['YEAR']))[0]
        D_res_test=D_res[(D_res['FLOW']>0) & (D_res['YEAR']==year)]
        if len(D_res_test)>2:
            D_res_optimal_filtered = D_res_optimal[D_res_optimal['YEAR']==year]
            
            
            
            #identifico il rettangolo da rappresentare
            min_lon = min(D_res_test[lonCol])
            max_lon= max(D_res_test[lonCol])
                
            min_lat=min(D_res_test[latCol])
            max_lat=max(D_res_test[latCol])
            
            #costruisco la griglia
            lon = np.linspace(min_lon,max_lon, 100)
            lat = np.linspace(min_lat, max_lat, 100)
            X, Y = np.meshgrid(lon, lat)
            xy_coord = list(zip(D_res_test[lonCol],D_res_test[latCol]))
            
            #interpolo la funzione nei punti mancanti
            grid = griddata(xy_coord, np.array(D_res_test['COST_TOBE']), (X, Y), method='linear')
            
            #salvo i valori per la rappresentazione 3d
            X_list.append(X)
            Y_list.append(Y)
            grid_list.append(grid)
            time_list.append(year)
            
            
            #se riceve un grafo stradale lo rappresenta
            if roadGraph==[]:
                fig1 = plt.figure()
                ax  = fig1.gca()
            else:
                fig1, ax = ox.plot_graph(roadGraph, bgcolor='k', 
                            node_size=1, node_color='#999999', node_edgecolor='none', node_zorder=2,
                            edge_color='#555555', edge_linewidth=0.5, edge_alpha=1)
                plt.legend(['Node','Edges'])
                
    
            
            
            im = ax.contour(X, Y, grid, cmap='Reds')
            
            ax.set_xlabel('LONGITUDE')
            ax.set_ylabel('LATITUDE')
            fig1.colorbar(im, ax=ax)
            
            
            ax.set_title(f"Isocost line {distanceType}, period: {year}")
            
            #rappresento i punti di ottimo
            ax.scatter(D_res_optimal_filtered[lonCol], D_res_optimal_filtered[latCol],100,marker='^',color='green')
            plt.legend(['Optimal points'])
            
            #rappresento i punti as-is
            if len(D_plant)>0:
                ax.scatter(D_plant[plantLongitude], D_plant[plantLatitude],100,marker='s',color='black')
                plt.legend(['Optimal points','Actual points'])
            
            fig1 = ax.figure
            outputFigure[f"isocost_{distanceType}_{year}"]=fig1
            
            plt.close('all')
    # costruisco il grafico 3d
    fig_curve_cost3D=createFigureWith3Dsurface(X_list,Y_list,grid_list,time_list)
    return outputFigure,fig_curve_cost3D


# %% CALCULATE THE SAVING
def calculateSaving(D_res_asis,
                    D_res_tobe,
                    periodCol_asis='YEAR',
                    periodCol_tobe='YEAR',
                    costCol_asis='COST_ASIS',
                    costCol_tobe='COST_TOBE',
                    titolo=''
                    ):
    
        output_figure={}
        df_saving=pd.DataFrame()
        
        D_saving_asis = D_res_asis.groupby(periodCol_asis)[costCol_asis].sum().to_frame()
        D_saving_tobe = D_res_asis.groupby(periodCol_tobe)[costCol_tobe].sum().to_frame()
        D_saving= D_saving_asis.merge(D_saving_tobe, how='left', left_on=periodCol_asis,right_on=periodCol_tobe)
        
        #D_saving= D_res_tobe.groupby(periodCol_asis)['COST_TOBE','COST_ASIS'].sum()
        D_saving['SAVING'] = D_saving[costCol_tobe]/D_saving[costCol_asis]
        fig1=plt.figure()
        plt.plot(D_saving.index,D_saving['SAVING'])
        plt.title(titolo)
        plt.xticks(rotation=45)
        
        
        output_figure['savingPercentage']=fig1
        df_saving=pd.DataFrame([np.mean(D_saving['SAVING'])])
        
        return output_figure, df_saving
    



# %%
'''
import matplotlib as mpl
import matplotlib.animation as animation
fig = plt.figure()
ims = []
for image in output_figures.values():
    ax = image.gca()
    imgs = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage)]
    #ims.append(image.gca().get_images())
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)    
    
# ani.save('dynamic_images.mp4')

plt.show()  
'''
# %% TEST aggiungere grafici con curve isoscosto
'''
from scipy.interpolate import RegularGridInterpolator
def f(x,y,z):
     return 2 * x**3 + 3 * y**2 - z
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))


a=np.meshgrid(x, y, z, indexing='ij', sparse=True)


# %% test righe isocosto
#https://plot.ly/python/lines-on-maps/
#https://plot.ly/python/3d-surface-plots/
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.griddata.html

#rectangularCost
    




D_filtered_notnan=D_filtered.dropna(subset=[latCol,lonCol,weightCol])
#identify limit points
min_x=math.trunc(min(D_filtered_notnan[lonCol]))
max_x=math.trunc(max(D_filtered_notnan[lonCol]))+1
min_y=math.trunc(min(D_filtered_notnan[latCol]))
max_y=math.trunc(max(D_filtered_notnan[latCol]))+1
min_z=math.trunc(min(D_filtered_notnan[weightCol]))
max_z=math.trunc(max(D_filtered_notnan[weightCol]))+1



# define meshgrid
x = np.linspace(min_x, max_x, 1000)
y = np.linspace(min_y, max_y, 1000)
z= np.linspace(min_z, max_z, 1000)
#xv, yv = np.meshgrid(x, y)


#interpolate the function wij X dig
data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))
data = zip(D_filtered_notnan[lonCol],D_filtered_notnan[latCol],D_filtered_notnan[weightCol])
data = list(zip(D_filtered_notnan[lonCol],D_filtered_notnan[latCol],D_filtered_notnan[weightCol]))
data_points=[]
for i,j,k in data: 
    data_points[i,j,k]=i,j,k
    
from scipy.interpolate import RegularGridInterpolator
my_interpolating_function = RegularGridInterpolator((x, y, z), data)
my_interpolating_function = RegularGridInterpolator((D_filtered_notnan[lonCol],D_filtered_notnan[latCol]),D_filtered_notnan[weightCol])

#define the surface
values = func_rectangularDistance(xv, yv, lon_optimal, lat_optimal)

import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=values)])
plotly.offline.plot(fig, filename = 'test.html', auto_open=True)
'''

