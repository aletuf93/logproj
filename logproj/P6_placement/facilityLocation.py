# -*- coding: utf-8 -*-

#import math
import pandas as pd
import numpy as np


#import plotly graphs

import plotly.graph_objects as go
import plotly.express as px

#import matplotlib colorbar
#from matplotlib import cm

from logproj.DIST.globalBookingAnalysis import getCoverageStats




# %% DEFINE FUNCTIONS FOR OPTIMAL LOCATION
def optimalLocationRectangularDistance(D_filtered, latCol, lonCol, weightCol):
    #the function returns the optimal location based on rectangular distances
    #D_filtered is a dataframe with columns LATITUDE, LONGITUDE and FLOWS indicating the location and the intensity of the flow
    
    #optimal location
    op_w=sum(D_filtered[weightCol])/2 # identify the median of the sum of weights
    
    #identify optimal latitude
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
    return lat_optimal, lon_optimal

def optimalLocationGravityProblem(D_filtered, latCol, lonCol, weightCol):
    #the dunction calculate the optimal location with squared euclidean distances 
    #D_filtered is a dataframe with flow intensity, latitude and longitude
    #latCol is a string with the name of the columns woith latitude
    #loncol is a string with the name of the columns with longitude
    #weightCol is a string with the name of the columns with the flow intensity
    
    D_filtered_notnan=D_filtered.dropna(subset=[latCol,lonCol,weightCol])
    lat_optimal=sum(D_filtered_notnan[latCol]*D_filtered_notnan[weightCol])/sum(D_filtered_notnan[weightCol])
    lon_optimal=sum(D_filtered_notnan[lonCol]*D_filtered_notnan[weightCol])/sum(D_filtered_notnan[weightCol])
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



# %% DEFINE FUNCTION TO CALCULATE THE OPTIMAL LOCATION
def calculateOptimalLocation(D_table,distanceType,latCol,lonCol,weightCol,descrCol,
                             D_plant=[], plantListName='LIST_NODE',plantLatitude='LATITUDE', plantLongitude='LONGITUDE'):
    
    # this function import a table D_table where each row is a node of the network
    #columns "NODE_DESCRIPTION" describe the node 
    #latCol identify the latitude of the node
    #lonCol identify the longitude of the node
    #weightCol is a dictionary where keys are years and values are flow values e.g. 2019:1000
    #descrCol is a column with description
    
    #D_plant is a dataframe containing the information on the plants ASIS
    # plantListName is the column name of the list with clients
    # plantLatitude is the column name of the latitude of the plant
    # plantLongitude is the column name of the longitude of the plant
    
    
    # it returns a dataframe D_res with the ID, LATITUDE, LONGITUDE AND YEAR
    # for each flow adding the column COST AND FLOW representing the distance 
    # travelled (COST) and the flow intensity (FLOW). The column
    # COST_NORM is a the flows scaled between 0 and 100
    
    # it returns a dataframe D_res_optimal with the loptimal latitude and longitude for each
    # time frame, and a column COST and FLOW with the total cost (distance) and flows
    
    
    #identifico gli anni nella colonna dizionario
    yearsColumns = list(D_table[weightCol].iloc[0].keys())
    
    
    # identifico le colonne utili
    D_res=pd.DataFrame(columns=[descrCol,latCol,lonCol,'YEAR','COST',])
    D_res_optimal=pd.DataFrame(columns=[descrCol,latCol,lonCol,'YEAR','COST','FLOW'])
    
    for year in yearsColumns:
        #year = yearsColumns[0]
        D_filter_columns=[descrCol,latCol,lonCol,year]
        D_filtered = D_table[D_filter_columns]
        D_filtered = D_filtered.rename(columns={year:'FLOW'})
        D_filtered['YEAR']=year
        
        # define optimal location
        if distanceType.lower()=='rectangular':
            lat_optimal, lon_optimal = optimalLocationRectangularDistance(D_filtered, latCol, lonCol, weightCol)
            D_filtered['COST']=func_rectangularDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered[weightCol])
        if distanceType.lower()=='gravity':
            lat_optimal, lon_optimal = optimalLocationGravityProblem(D_filtered, latCol, lonCol, weightCol)
            D_filtered['COST']=func_gravityDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered[weightCol])
        elif distanceType.lower()=='euclidean':
            lat_optimal, lon_optimal = optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, weightCol)
            D_filtered['COST']=func_euclideanDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered[weightCol])
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
    
    #calcolo le coperture
    D_res_null = D_res
    D_res_null=D_res_null.replace({0:np.nan})
    accuracy , _ = getCoverageStats(D_res_null,'FLOW','FLOW_norm') #non considerare il secondo valore di accuratezza
    D_res['accuracy']=[accuracy for i in range (0,len(D_res))]
    
    if len(D_plant)>0:
        # a questo punto se ho le coordinate calcolo le distanze AS-IS con i plant attuali
        D_res=D_res.rename(columns={'COST':'COST_TOBE'})
        D_res['COST_ASIS']=np.nan
        D_res=D_res.reset_index(drop=True)
        #columnsSol=list(D_res.columns)
        #columnsSol.append('COST_ASIS_EUCLIDEAN')
        
        
        #D_sol=pd.DataFrame(columns=columnsSol)
        for plant in D_plant['_id']:
            #plant =652
            
            #identifico lista clienti, latitudine e longitudine
            plant_client_list = D_plant[D_plant['_id']==plant][plantListName].iloc[0]
            lat_plant= D_plant[D_plant['_id']==plant][plantLatitude].iloc[0]
            lon_plant= D_plant[D_plant['_id']==plant][plantLongitude].iloc[0]
            
            #creo una nuova colonna per identificare se servito da un determinato plant
            D_res[plant]= [id_nodo in plant_client_list for id_nodo in D_res['_id']]
            
            #D_res['all']=D_res[652].astype(int) + D_res[2615].astype(int) +D_res[603].astype(int) + D_res[610].astype(int)
            D_filtered = D_res[D_res[plant]==True]
            idx_to_upload = D_filtered.index.values
            #continuare da qui
            distancecost = list(func_rectangularDistanceCost(D_filtered[lonCol], 
                                                                           D_filtered[latCol], 
                                                                           lon_plant, 
                                                                           lat_plant, 
                                                                           D_filtered['FLOW'])
                                )
            for i in range(0,len(distancecost)):
                D_res['COST_ASIS'].loc[idx_to_upload[i]] = distancecost[i]
    
    return D_res, D_res_optimal

# %% DEFINE FUNCTIONS TO GENERATE GRAPHS
    
def createFigureBubbleFlows(D_res,latCol,lonCol,weightCol,descrCol):
    #D_res is a dataframe with flows defined by the function calculateOptimalLocation
    #latCol is a string with column name for latitude
    #lonCol is a string with column name for longitude
    #weightCol is a string with column name for flow intensity
    #descrCol is a string with column name for node description
    fig_geo = px.scatter_mapbox(D_res, 
                     lat =latCol,
                     lon=lonCol,
                     color="COST",
                     hover_name=descrCol, 
                     size=weightCol,
                     animation_frame="YEAR",
                        )
                     #projection="natural earth")
    fig_geo.update_layout(mapbox_style="open-street-map")
    return fig_geo

def createFigureOptimalPoints(D_res_optimal,latCol,lonCol,descrCol):
    #D_res is a dataframe with flows defined by the function calculateOptimalLocation
    #latCol is a string with column name for latitude
    #lonCol is a string with column name for longitude
    #descrCol is a string with column name for node description
    
   

    # define the optimal location
    fig_optimal = px.line_mapbox(D_res_optimal, 
                             lat=latCol, 
                             lon=lonCol, 
                             #animation_frame="YEAR", 
                             hover_name=descrCol,
                             #mode = 'lines'
                             #color='COLOR',
                             #color_continuous_scale='Inferno',
                             
                             )
    fig_optimal.update_layout(mapbox_style="open-street-map")          
    return fig_optimal




def createFigureWithOptimalPointsAndBubbleFlows(D_res,D_res_optimal,latCol,lonCol,descrCol):
    
    #D_res is a dataframe with flows defined by the function calculateOptimalLocation
    #D_res_optimal is a dataframe with optimal locations defined by the function calculateOptimalLocation
    #latCol is a string with column name for latitude
    #lonCol is a string with column name for longitude
    #descrCol is a string with column name for node description
    
    ########################################
    ######### DEFINE RAW FIGURE ############
    ########################################
    #define raw figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    
    
    ########################################
    ######### DEFINE BASE LAYOUT ###########
    ########################################
    
    # Identify all the years
    years= list(D_res_optimal["YEAR"].sort_values())
    
    # define layout hovermode
    fig_dict["layout"]["hovermode"] = "closest"
    
    # define sliders
    fig_dict["layout"]["sliders"] = {
        "args": [
            "transition", {
                "duration": 400,
                "easing": "cubic-in-out"
            }
        ],
        "initialValue": "1952",
        "plotlycommand": "animate",
        "values": years,
        "visible": True
    }
        
    #define menus and buttons
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    
    #define slider dictionary
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Year:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    
    ########################################
    ######### DEFINE DATA ##################
    ########################################
    
    
    
    
    #start from the first frame and define the figure
    year = years[0]
    
    
    
    ########################################
    ######### DEFINE FIGURE ################
    ########################################
    
    #define the trace with optimal point
    currentColor=0
    
    
        
    data_dict = go.Scattermapbox(
                        lat=D_res_optimal[D_res_optimal['YEAR']==year][latCol],
                        lon=D_res_optimal[D_res_optimal['YEAR']==year][lonCol],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=14,
                            #color='red',
                            #color=currentColor,
                            #color=cm.Reds(colore),
                            color=[i for i in range(0,len(D_res_optimal['YEAR']==year))],
                            opacity=1,
                            colorscale="Reds"
                        ),
                        text=D_res_optimal[D_res_optimal['YEAR']==year][descrCol],
                        name='optimal'
                        )
                        
    
    
    fig_dict["data"].append(data_dict)
    
    #define the trace with bubbles of the other flows
    data_dict = go.Scattermapbox( 
                         lat =D_res[D_res['YEAR']==year][latCol],
                         lon=D_res[D_res['YEAR']==year][lonCol],
                         #color=,
                         text=D_res[D_res['YEAR']==year][descrCol],
                         marker=go.scattermapbox.Marker(
                            size=D_res[D_res['YEAR']==year]['FLOW_norm'],
                            color=D_res[D_res['YEAR']==year]['COST_TOBE'],
                            opacity=0.5,
                            showscale=True,
                            colorscale='Viridis',
                            ),
                         name = 'flow intensity'
                    )
                         #projection="natural earth")
    fig_dict["data"].append(data_dict)
    
    
    ########################################
    ######### DEFINE FRAMES ################
    ########################################
    
    for year in years:
        frame = {"data": [], "name": year}
        
        #count the current color to have a gradient in the optimal point
        currentColor=currentColor+1
        
        
        
        #define the trace with optimal point
        data_dict = go.Scattermapbox(
                        lat=D_res_optimal[D_res_optimal['YEAR']<=year][latCol],
                        lon=D_res_optimal[D_res_optimal['YEAR']<=year][lonCol],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=14,
                            #color='red',
                            #color=currentColor,
                            #color=cm.Reds(colore),
                            color=[i for i in range(0,len(D_res_optimal['YEAR']==year))],
                            opacity=1,
                            colorscale="Reds"
                        ),
                        text=D_res_optimal[D_res_optimal['YEAR']<=year][descrCol],
                        name='optimal'
                        )
        frame["data"].append(data_dict)
        
        
        #define the trace with bubbles of the other flows
        data_dict = go.Scattermapbox( 
                             lat =D_res[D_res['YEAR']==year][latCol],
                             lon=D_res[D_res['YEAR']==year][lonCol],
                             #color=,
                             text=D_res[D_res['YEAR']==year][descrCol],
                             marker=go.scattermapbox.Marker(
                                size=D_res[D_res['YEAR']==year]['FLOW_norm'],
                                color=D_res[D_res['YEAR']==year]['COST_TOBE'],
                                opacity=0.5,
                                showscale=True,
                                colorscale='Viridis',
                                ),
                             name='flow intensity'
                        )
                             #projection="natural earth")
        frame["data"].append(data_dict)   
        fig_dict["frames"].append(frame)
        
        
        # update the slider
        slider_step = {"args": [
            [year],
            {"frame": {"duration": 300, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": year,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
    
    #update the layout
    fig_dict["layout"]["sliders"] = [sliders_dict]
    #create the figure
    fig = go.Figure(fig_dict)
    
    #update with openStreetMap style
    fig.update_layout(mapbox_style="open-street-map")  
    return fig









