# -*- coding: utf-8 -*-

#import math
import pandas as pd
import numpy as np
import sys

root_folder="C:\\Users\\aletu\\Documents\\GitHub\\production"
sys.path.append(root_folder)

root_folder="C:\\Users\\aletu\\Documents\\GitHub"
sys.path.append(root_folder)

#import plotly graphs
import plotly
import plotly.graph_objects as go
import plotly.express as px


# import packages
import database.mongo_encrypt as menc
import database.mongo_loginManager as mdb
import database.odm_production_mongo as model


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
def calculateOptimalLocation(D_table,distanceType,latCol,lonCol,weightCol,descrCol):
    
    # this function import a table D_table where each row is a node of the network
    #columns "NODE_DESCRIPTION" describe the node 
    #LATITUDE identify the latitude of the node
    #LONGITUDE identify the longitude of the node
    #FLOW is a dictionary where keys are years and values are flow values e.g. 2019:1000
    
    
    #identifico gli anni nella colonna dizionario
    yearsColumns = list(D_table[weightCol][0].keys())
    
    # identifico le colonne utili
    D_res=pd.DataFrame(columns=[descrCol,latCol,lonCol,'YEAR','COST',])
    D_res_optimal=pd.DataFrame(columns=[descrCol,latCol,lonCol,'YEAR'])
    
    for year in yearsColumns:
        #year = yearsColumns[0]
        D_filter_columns=[descrCol,latCol,lonCol,year]
        D_filtered = D[D_filter_columns]
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
        
        
        D_res_optimal=D_res_optimal.append(pd.DataFrame([[f"OPTIMAL LOCATION YEAR: {year}",lat_optimal, lon_optimal, year]], columns=D_res_optimal.columns))
    
        
    #D_res['COST_norm']=(D_res['COST']-min(D_res['COST']))/(max(D_res['COST'])-min(D_res['COST']))*10
    D_res['FLOW_norm']=(D_res['FLOW']-min(D_res['FLOW']))/(max(D_res['FLOW'])-min(D_res['FLOW']))*100
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

def createFigureOptimalPoints(D_res,latCol,lonCol,descrCol):
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
    data_dict = go.Scattermapbox(
                        lat=D_res_optimal[D_res_optimal['YEAR']==year][latCol],
                        lon=D_res_optimal[D_res_optimal['YEAR']==year][lonCol],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=14,
                            color='red',
                            opacity=0.8
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
                            color=D_res[D_res['YEAR']==year]['COST'],
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
        
        #define the trace with optimal point
        data_dict = go.Scattermapbox(
                        lat=D_res_optimal[D_res_optimal['YEAR']<=year][latCol],
                        lon=D_res_optimal[D_res_optimal['YEAR']<=year][lonCol],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=14,
                            color='red',
                            opacity=0.8
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
                                color=D_res[D_res['YEAR']==year]['COST'],
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

# %% IMPORT data from mongodb
    
# connect to the database
caseStudy_string='CHIMAR_5'
latCol='LATITUDE'
lonCol='LONGITUDE'
weightCol='FLOW'
descrCol='NODE_DESCRIPTION'

# import data
dbName=menc.encryptString(caseStudy_string)
mdb.setConnection(dbName)
D_table=mdb.queryTodf(model.node.objects)
D = D_table.join(D_table['FLOW'].apply(pd.Series))

#simulate scenarios with different distance types
for distanceType in ['euclidean','gravity','rectangular']:
    # calculate optimal location
    D_res, D_res_optimal=calculateOptimalLocation(D_table,distanceType,latCol,lonCol,weightCol,descrCol)

    #define figure
    fig = createFigureWithOptimalPointsAndBubbleFlows(D_res,D_res_optimal,latCol,lonCol,descrCol)
    #fig = createFigureBubbleFlows(D_res,latCol,lonCol,weightCol,descrCol)
    #fig = createFigureOptimalPoints(D_res,latCol,lonCol,descrCol)
    plotly.offline.plot(fig, filename = f"facilityLocation_{distanceType}.html", auto_open=True)





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

