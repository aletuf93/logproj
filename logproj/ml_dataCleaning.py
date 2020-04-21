# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.special import erfc

# %% DATA CLEANING
def chauvenet(y, mean=None, stdv=None):
   #-----------------------------------------------------------
   # Input:  NumPy arrays x, y that represent measured data
   #         A single value of a mean can be entered or a
   #         sequence of means with the same length as
   #         the arrays x and y. In the latter case, the
   #         mean could be a model with best-fit parameters.
   # Output: It returns a boolean array as filter.
   #         The False values correspond to the array elements
   #         that should be excluded
   #
   # First standardize the distances to the mean value
   # d = abs(y-mean)/stdv so that this distance is in terms
   # of the standard deviation.
   # Then the  CDF of the normal distr. is given by
   # phi = 1/2+1/2*erf(d/sqrt(2))
   # Note that we want the CDF from -inf to -d and from d to +inf.
   # Note also erf(-d) = -erf(d).
   # Then the threshold probability = 1-erf(d/sqrt(2))
   # Note, the complementary error function erfc(d) = 1-erf(d)
   # So the threshold probability pt = erfc(d/sqrt(2))
   # If d becomes bigger, this probability becomes smaller.
   # If this probability (to obtain a deviation from the mean)
   # becomes smaller than 1/(2N) than we reject the data point
   # as valid. In this function we return an array with booleans
   # to set the accepted values.
   #
   # use of filter:
   # xf = x[filter]; yf = y[filter]
   # xr = x[~filter]; yr = y[~filter]
   # xf, yf are cleaned versions of x and y and with the valid entries
   # xr, yr are the rejected values from array x and y
   #-----------------------------------------------------------
   if mean is None:
      mean = y.mean()           # Mean of incoming array y
   if stdv is None:
      stdv = y.std()            # Its standard deviation
   N = len(y)                   # Lenght of incoming arrays
   criterion = 1.0/(2*N)        # Chauvenet's criterion
   d = abs(y-mean)/stdv         # Distance of a value to mean in stdv's
   d /= 2.0**0.5                # The left and right tail threshold values
   prob = erfc(d)               # Area normal dist.
   filter = prob >= criterion   # The 'accept' filter array with booleans
   return filter                # Use boolean array outside this function

# In[1]: CLEANING FUNCTIONS

# Pulizia dei dati. Devo andare ad applicare il criterio di Chauvenet sulle features per eliminare quelle assurde!
def cleanOutliers(tableSKU, features): #applica il criterio di chauvenet colonna per colonna e restituisce la tabella pulita

    good=np.ones(len(tableSKU))

    for i in range(0,len(features)):
        temp=tableSKU.loc[:,features[i]]
        values=chauvenet(temp)
        good=np.logical_and(good,values)

    #df = tableSKU.loc[good, features]
    df = tableSKU[good]
    #itemcodeOK=tableSKU.loc[good,'itemcode']

    Perc=np.around(float(len(df))/len(tableSKU)*100,2) #è la percentuale di dati buoni
    return df, Perc

# Clean data using the interquartile range method
def cleanUsingIQR(table, features,capacityField=[]):
    
    '''
    use IQR method to clean a datarame table
    data cleaning is applied on the list features, one feature at a time
    capacityField is the column name of a quantity metric of table to compute the coverage
    
    it returns the cleaned dataset temp and a coerage tuple
    '''
    table=temp=table.reset_index(drop=True)
    for feature in features:
        
        if len(temp[feature])>0:
            q1, q3= np.percentile(temp[feature].dropna(),[25,75]) #percentile ignoring nan values
            if (q1!=None) & (q3!=None):
                iqr = q3 - q1
                lower_bound = q1 -(1.5 * iqr)
                upper_bound = q3 +(1.5 * iqr)
                temp=temp[(temp[feature]<=upper_bound) & (temp[feature]>=lower_bound)]
                temp=temp.reset_index(drop=True)
    lineCoverage = len(temp)/len(table)
    qtyCoverage=np.nan
    if len(capacityField)>0:
        qtyCoverage = np.nansum(temp[capacityField])/np.nansum(table[capacityField])
          
    return temp, (lineCoverage,qtyCoverage)


# %% DATA PREPROCESSING

#se sono rimaste delle variabili categoriche nel dataframe X le trasformo in dummy
def dummyColumns(X):
    #X is a dataframe
    #The index of the dataframe is not modified
    try:
        for column in X.columns:
         try:
             if X[column].dtype==object:
                 dummyCols=pd.get_dummies(X[column])
                 X=pd.concat([X,dummyCols], axis=1)
                 del X[column]
         except:
             True
    except:
        True
    return X

#usa trasformazione seno e coseno per un campo orario
def transformClockData(series):
    '''
    use cosine and sine transformation to a series 
    (e.g. indicating the hour of the days, or the minutes)
    '''
    transformedDataCos=np.cos(2*np.pi*series/max(series))
    transformedDataSin=np.sin(2*np.pi*series/max(series))
    return transformedDataCos, transformedDataSin