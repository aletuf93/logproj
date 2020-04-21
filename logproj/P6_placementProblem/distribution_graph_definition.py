# -*- coding: utf-8 -*-

#https://github.com/gboeing/osmnx-examples/tree/master/notebooks
import database.mongo_loginManager as mdb
import database.models.odm_distribution_mongo as model_dist
import database.models.odm_production_mongo as model_prod

from logproj.ml_dataCleaning import cleanUsingIQR

import osmnx as ox
import numpy as np
import pandas as pd


def import_graph_drive(dbName,latCol,lonCol, plantLatitude, plantLongitude,cleanOutliers=False):
    
    '''
    the function imports a road network using osmnx library
    dbName is the name of the collection of mongodb to use
    latCol is the name attribute of the latitude of the node collection
    lonCol is the name attribute of the longitude of the node collection
    plantLatitude is the name attribute of the latitude of the plant collection
    plantLongitude is the name attribute of the longitude of the plant collection
    cleanOutliers is True to remove outliers of latitude and logitude by using IQR
    
    return the cleaned dataframe and a coverage tuple
    
    '''
    
    coverages=(1,np.nan)
    mdb.setConnection(dbName)
    D_plant=mdb.queryTodf(model_prod.plant.objects)
    
    
    
    
    D_node=mdb.queryTodf(model_dist.node.objects)
    
    #remove latitude and longitude outliers
    if cleanOutliers:
          D_node, coverages, =cleanUsingIQR(D_node, [latCol,lonCol])
    
    allLatitudes=list(D_node[latCol]) + list(D_plant[plantLatitude])
    allLongitudes=list(D_node[lonCol]) + list(D_plant[plantLongitude])
    
    min_lat = min(allLatitudes)
    max_lat = max(allLatitudes)
    min_lon = min(allLongitudes)
    max_Lon = max(allLongitudes)
    
    G = ox.graph_from_bbox(max_lat, min_lat,max_Lon,min_lon, network_type='drive')
    
    output_coverages = pd.DataFrame(coverages)
    return G, output_coverages




'''
G = ox.graph_from_place('Friuli Venezia giulia, Italy', network_type='drive')
#G = ox.simplify_graph(G)



fig, ax = ox.plot_graph(G, bgcolor='k', 
                        node_size=1, node_color='#999999', node_edgecolor='none', node_zorder=2,
                        edge_color='#555555', edge_linewidth=0.5, edge_alpha=1)



#%% LESSON 1

# Import necessary geometric objects from shapely module
from shapely.geometry import Point, LineString, Polygon

# Create Point geometric object(s) with coordinates
point1 = Point(2.2, 4.2)
point2 = Point(7.2, -25.1)
point3 = Point(9.26, -2.456)
point3D = Point(9.26, -2.456, 0.57)

# What is the type of the point?
point_type = type(point1)

print(point1)
print(point3D)
print(type(point1))

# get coordinates
point1.x
point1.y
point1.xy

# %% LESSON 2
import geopandas as gpd

# Set filepath
fp = "L2_data//L2_data//Europe_borders.shp"

# Read file using gpd.read_file()
data = gpd.read_file(fp)

type(data)
data.plot()

#get the coordinates reference system
data.crs

#chenage crs
data2 = data.to_crs(epsg=3035)
data2.plot()


#%% LESSON 3
from geopandas.tools import geocode
import pandas as pd
# Filepath
fp = "L3_data//L3_data//addresses.txt"

# Read the data
data = pd.read_csv(fp, sep=';')

# Geocode addresses with Nominatim backend (nominatim=openstreetmap)
geo = geocode(data['addr'], provider='nominatim', user_agent='csc_user_ht')

join = geo.join(data)
join.head()
join.plot()


#retrieve data with osmnx
import osmnx as ox

# Specify the name that is used to seach for the data
place_name = "Kamppi, Helsinki, Finland"

# Fetch OSM street network from the location
graph = ox.graph_from_place(place_name)
type(graph)

#plot graph
fig, ax = ox.plot_graph(graph)

# Retrieve the footprint of our location
area = ox.gdf_from_place(place_name)
area.plot()
# Retrieve buildings from the area
buildings = ox.buildings_from_place(place_name)

# What types are those?
print(type(area))
print(type(buildings))


#data reclassification
#https://automating-gis-processes.github.io/CSC/notebooks/L3/reclassify.html

#spatial weigths
#https://nbviewer.jupyter.org/github/pysal/splot/blob/master/notebooks/libpysal_non_planar_joins_viz.ipynb
'''