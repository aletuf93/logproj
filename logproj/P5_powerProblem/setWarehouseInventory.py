# -*- coding: utf-8 -*-
import numpy as np
def returnStockoutRisk(x,a,b,c):
    '''
    returns the CDF value of a triangular distribution

    Parameters
    ----------
    x : TYPE float
        DESCRIPTION. independent variable of the CDF
    a : TYPE float 
        DESCRIPTION. min value of the triangular distribution
    b : TYPE float
        DESCRIPTION. max value of the triangular sitribution
    c : TYPE float
        DESCRIPTION. mode of the triangular distribution

    Returns
    -------
    TYPE float
        DESCRIPTION. value of risk associated with the inventory level x

    '''
    probability = np.nan
    if x<=a:
        probability= 0
    elif (a<x) & (x<=c):
        probability= ((x-a)**2)/((b-a)*(c-a)) 
    elif (c<x) & (x<b):
        probability= 1 - ((b-x)**2)/((b-a)*(b-c))
    elif (x>=b):
        probability= 1
    else:
        print("Error in the CDF")
    return 1-probability


def returnStockQuantityFromRisk(risk,a,b,c):
    '''
    

    Parameters
    ----------
    risk : TYPE float
        DESCRIPTION. value of risk associated to the probability distribution (inverse of the probability)
     a : TYPE float 
        DESCRIPTION. min value of the triangular distribution
    b : TYPE float
        DESCRIPTION. max value of the triangular sitribution
    c : TYPE float
        DESCRIPTION. mode of the triangular distribution

    Returns
    -------
    x : TYPE float
        DESCRIPTION. value of inventory associated with the risk 

    '''
    

    u = 1-risk
    x = np.nan
    if (0<=u) & (u<((c-a)/(b-a))) :
        x= a + np.sqrt((b-a)*(c-a)*u)
    elif (((c-a)/(b-a))<=u) & (u<=1):
        x = b - np.sqrt((b-a)*(b-c)*(1-u))
    
    else:
        print("Error in the CDF")
    return x