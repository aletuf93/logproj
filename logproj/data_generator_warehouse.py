# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:30:38 2020

@author: aletu
"""
import numpy as np
import pandas as pd
import random
import datetime


def generateWarehouseData(num_SKUs = 100,
    nodecode = 1,
    idwh = ['LOGICAL_WH1', 'LOGICAL_WH2', 'FAKE'],
    whsubarea = ['AREA 1'],
    num_corsie = 5,
    num_campate = 66,
    num_livelli = 5,
    alt_livello = 1200,
    largh_campate = 800,
    largh_corsia = 4000,
    num_movements=1000,
    num_ordercode = 800,
    average_time_between_movements = 1/24, #days
    first_day = datetime.datetime(year=2020, month=1, day = 2),
    ):

    
    #% CLASS SKU
    class SKU():
        
        def __init__(self,itemcode):
            self.ITEMCODE=itemcode
            self.DESCRIPTION = f"PRODOTTO_{itemcode}"
            self.VOLUME = np.random.uniform(0.1,100) #volume in dm3
            self.WEIGHT = np.random.uniform(0.1,10) #weigth in Kg
       
     
    #% CLASS STORAGE LOCATION
    class STORAGE_LOCATION():
        
        def __init__(self,nodecode, idwh, whsubarea, idlocation, loccodex, loccodey, loccodez, rack, bay, level):
            
            self.NODECODE = nodecode
            self.IDWH = idwh
            self.WHSUBAREA = whsubarea
            self.IDLOCATION=idlocation
            self.LOCCODEX = loccodex
            self.LOCCODEY = loccodey
            self.LOCCODEZ = loccodez
            self.RACK = rack
            self.BAY = bay 
            self.LEVEL = level
            
            
    # % CLASS MOVEMENTS 
    class MOVEMENTS():
        def __init__(self, itemcode,volume,weight,  nodecode, idwh, whsubarea, idlocation,rack, bay, level, loccodex, loccodey, loccodez, 
                     ordercode, quantity, timestamp, inout, ordertype):
            self.ITEMCODE=itemcode
            
            self.NODECODE = nodecode
            self.IDWH = idwh
            self.WHSUBAREA = whsubarea
            self.IDLOCATION=idlocation
            self.RACK=rack 
            self.BAY=bay 
            self.LEVEL=level
            self.LOCCODEX = loccodex
            self.LOCCODEY = loccodey
            self.LOCCODEZ = loccodez
            
            self.ORDERCODE = ordercode
            self.PICKINGLIST = ordercode
            self.QUANTITY = quantity
            self.VOLUME = volume*quantity
            self.WEIGHT = weight*quantity
            self.TIMESTAMP_IN = timestamp
            self.INOUT = inout
            self.ORDERTYPE = ordertype
            
    # % CLASS INVENTORY
    class INVENTORY():
        def __init__(self,itemcode,nodecode, idwh, idlocation,quantity, timestamp):
            self.NODECODE = nodecode
            self.IDWH=idwh
            self.ITEMCODE=itemcode
            self.IDLOCATION = idlocation
            self.QUANTITY = quantity
            self.TIMESTAMP = timestamp
            
    #% CREATE SKUS
    
    dict_SKUs={}
    itemcodes=np.arange(0,num_SKUs)
    for itemcode in itemcodes:
        dict_SKUs[itemcode] = SKU(itemcode)
        
        
    # % CREATE WH LAYOUT
    dict_locations ={}
    idlocation=0
    
    for corsia in range(0, num_corsie):
        for campata in range(0, num_campate):
            for livello in range(0,num_livelli):
                idlocation=idlocation+1 #create a new location index
                
                #save parameters
                NODECODE = nodecode
                IDWH = random.choice(idwh)
                WHSUBAREA = random.choice(whsubarea) 
                IDLOCATION = idlocation
                LOCCODEX = corsia*largh_corsia
                LOCCODEY = campata*largh_campate
                LOCCODEZ = livello*alt_livello
                
                #create storage location
                dict_locations[idlocation] = STORAGE_LOCATION(NODECODE, 
                                                              IDWH, 
                                                              WHSUBAREA, 
                                                              IDLOCATION, 
                                                              LOCCODEX, 
                                                              LOCCODEY, 
                                                              LOCCODEZ, 
                                                              corsia, 
                                                              campata, 
                                                              livello)
             
    # %% CREATE MOVEMENTS
    
    dict_movements={}
    num_creati = 0
    ordercodes = np.arange(0,num_ordercode)
    
    while num_creati < num_movements:
        num_creati = num_creati+1
        
        
        #random select sku
        sku = random.choice(dict_SKUs)
        itemcode = sku.ITEMCODE
        volume = sku.VOLUME 
        weight = sku.WEIGHT
        
        #random select storage location
        loc_key = random.choice(list(dict_locations.keys()))
        loc = dict_locations[loc_key]
        nodecode = loc.NODECODE
        idwh = loc.IDWH
        whsubarea=loc.WHSUBAREA
        idlocation=loc.IDLOCATION
        loccodex = loc.LOCCODEX
        loccodey = loc.LOCCODEY  
        loccodez = loc.LOCCODEZ
        rack = loc.RACK 
        bay=loc.BAY 
        level = loc.LEVEL
        
        #generates movements data
        ordercode = random.choice(ordercodes)
        quantity = np.random.lognormal(mean=2,sigma=1)
        wait = np.random.exponential(average_time_between_movements)
        if num_creati==1:
            timestamp = first_day + datetime.timedelta(wait)
        else:
            timestamp = dict_movements[num_creati-1].TIMESTAMP_IN + datetime.timedelta(wait)
        
        inout = random.choice(['+','-',' '])
        ordertype = random.choice(['PICKING','PUTAWAY',' OTHER '])
        dict_movements[num_creati] = MOVEMENTS(itemcode,volume,weight , nodecode, idwh, whsubarea, idlocation,rack, bay, level, loccodex, loccodey, loccodez, 
                     ordercode, quantity, timestamp, inout, ordertype)
        
    # %% CREATE INVENTORY
    dict_inventory = {}
    for itemcode in dict_SKUs:
        #sku = dict_SKUs[itemcode]
        
        loc_key = random.choice(list(dict_locations.keys()))
        loc = dict_locations[loc_key]
        nodecode=loc.NODECODE
        idwh=loc.IDWH
        idlocation=loc.IDLOCATION
        quantity = np.random.lognormal(mean=2,sigma=1)
        dict_inventory[itemcode] = INVENTORY(itemcode,nodecode, idwh, idlocation,quantity, first_day)
        
        
        
    # %% SAVE LOCATIONS AND EXPORT
    
    D_locations = pd.DataFrame()
    for loc in dict_locations:
        D_locations=D_locations.append(pd.DataFrame([vars(dict_locations[loc])])) 
    #D_locations.to_excel('ubiche.xlsx')
    
    # %% SAVE SKUS
    
    D_SKUs = pd.DataFrame()
    for sku in dict_SKUs:
        D_SKUs=D_SKUs.append(pd.DataFrame([vars(dict_SKUs[sku])]))
    #D_SKUs.to_excel('anagrafica.xlsx')
    
    # %% SAVE MOVEMENTS
    
    D_movements = pd.DataFrame()
    for mov in dict_movements:
        D_movements=D_movements.append(pd.DataFrame([vars(dict_movements[mov])]))
    #D_movements.to_excel('movimenti.xlsx')
    
    # %% SAVE INVENTORY 
    D_inventory = pd.DataFrame()
    for inv in dict_inventory:
        D_inventory=D_inventory.append(pd.DataFrame([vars(dict_inventory[inv])]))
    
    return D_locations, D_SKUs, D_movements, D_inventory

# %% degub area
#D_locations, D_SKUs, D_movements, D_inventory = generateWarehouseData()

#D_movements.to_excel('movimenti.xlsx')
#D_SKUs.to_excel('anagrafica.xlsx')
#D_locations.to_excel('ubiche.xlsx')
#D_inventory.to_excel('giacenza.xlsx')
