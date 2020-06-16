# -*- coding: utf-8 -*-

# %% geocode
import geocoder
import time

mykey = '' #insert here your bing API key
def bruteforceGeocoding(dict_address,apiKey,waitTime):
    #la funziona tenta un approccio di georeferenziazione a forza bruta prendendo
    #prima tutti i dati a disposizione e poi andando a scalare se questi
    #non danno risultati 
    
    
    time.sleep(waitTime)
    if ('ADDRESS' in dict_address.keys()) & ('CITY' in dict_address.keys()) & ('ZIPCODE' in dict_address.keys()):
        g = geocoder.bing(location=None, addressLine=dict_address['ADDRESS'], locality=dict_address['CITY'], postalCode=dict_address['ZIPCODE'],  method='details', key=apiKey)
        a = g.json
        if a is not None: return 
    
    time.sleep(waitTime)
    if ('ADDRESS' in dict_address.keys()) & ('CITY' in dict_address.keys()):
        g = geocoder.bing(location=None, addressLine=dict_address['ADDRESS'], locality=dict_address['CITY'], method='details', key=apiKey)
        a = g.json
        if a is not None: return a
    
    time.sleep(waitTime)
    if ('CITY' in dict_address.keys()):
        g = geocoder.bing(location=None, locality=dict_address['CITY'],  method='details', key=apiKey)
        a = g.json
        if a is not None: return a
    
    #se sono arrivato qui non ho trovato nulla    
    return None    
        
        #aspetto un secondo per non impallare bing
        


#%% geocoding function
def addressGeocoder(dict_address, apiKey=mykey,waitTime=1):
    #the function get the geo info given a geo input 
    #dict address is a dictionary with geo info (e.g. name, address, zipcode, etc.)
    
    #wait one sec
    
    
    
 
         
        
    #query bing 
    a = bruteforceGeocoding(dict_address,apiKey,waitTime)
          
    result ={}
    if a is None:
        print("**GEOCODER: No geodata found using BING")
        return []
    else:
        
        if 'lat' in a.keys():
            result['LATITUDE_api'] = a['lat'] 
        
        if 'lng' in a.keys():
            result['LONGITUDE_api'] = a['lng']
        
        if 'address' in a.keys():
            result['ADDRESS_api'] = a['address']
        
        if 'city' in a.keys():
            result['CITY_api'] = a['city']
        
        if 'country' in a.keys():
            result['COUNTRY_api'] = a['country']
        
        if 'state' in a.keys():
            result['STATE_api'] = a['state']
        
        if 'postal' in a.keys():
            result['ZIPCODE_api'] = a['postal'] 
        
        print (f"**GEOCODER: geodata found at {result}")
        return result
# %% directGeocoder
def directGeocoder(dict_geo, apiKey=mykey,waitTime=1):
    #use a dictionary with latitude and longitude to find information on the geopoint
    result={'LATITUDE_api':dict_geo['LATITUDE'],
            'LONGITUDE_api':dict_geo['LONGITUDE'],
            }
    
    time.sleep(waitTime)
    if ('LATITUDE' in dict_geo.keys()) & ('LONGITUDE' in dict_geo.keys()):
        g = geocoder.bing([dict_geo['LATITUDE'], dict_geo['LONGITUDE']], method='reverse', key=apiKey)
        a = g.json
    if a is None:
        print("**GEOCODER: No geodata found using BING")
        return []
    else:
        
        if 'lat' in a.keys():
            result['LATITUDE_api'] = a['lat'] 
        
        if 'lng' in a.keys():
            result['LONGITUDE_api'] = a['lng']
        
        if 'address' in a.keys():
            result['ADDRESS_api'] = a['address']
        
        if 'city' in a.keys():
            result['CITY_api'] = a['city']
        
        if 'country' in a.keys():
            result['COUNTRY_api'] = a['country']
        
        if 'state' in a.keys():
            result['STATE_api'] = a['state']
        
        if 'postal' in a.keys():
            result['ZIPCODE_api'] = a['postal'] 
        
        print (f"**GEOCODER: geodata found at {result}")
        return result
    

    