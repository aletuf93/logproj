# -*- coding: utf-8 -*-

#https://github.com/gboeing/osmnx-examples/tree/master/notebooks
#import database.mongo_loginManager as mdb
#import database.models.odm_distribution_mongo as model_dist
#import database.models.odm_production_mongo as model_prod

from logproj.ml_dataCleaning import cleanUsingIQR

import osmnx as ox
import numpy as np
import pandas as pd


def import_graph_drive(D_node,latCol,lonCol,D_plant, plantLatitude, plantLongitude,cleanOutliers=False):
    
    '''
    the function imports a road network using osmnx library
    
    D_node is the table containing the nodes of the network
    latCol is the name attribute of the latitude of the node collection
    lonCol is the name attribute of the longitude of the node collection
    
    D_plant id the table containing the plant of the network
    plantLatitude is the name attribute of the latitude of the plant collection
    plantLongitude is the name attribute of the longitude of the plant collection
    cleanOutliers is True to remove outliers of latitude and logitude by using IQR
    
    return the cleaned dataframe and a coverage tuple
    
    '''
    
    coverages=(1,np.nan)
    #mdb.setConnection(dbName)
    #D_plant=mdb.queryTodf(model_prod.plant.objects)
    
    
    
    
    #D_node=mdb.queryTodf(model_dist.node.objects)
    
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



