# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logproj.stat_time_series as ts
from statsmodels.tsa.seasonal import seasonal_decompose
# %% PLOT TRENDS WITH SUBPLOTS
    
def plotquantitytrend(D_temp,  date_field='TIMESTAMP_IN', filterVariable=[], filterValue=[],  quantityVariable = 'sum_QUANTITY', countVariable = 'count_TIMESTAMP_IN', titolo=''):
    #the function return a figure with two subplots on for quantities the other for lines
    # D_temp is the input dataframe
    #data_fiels is the string with the column name for the date field
    # filterVariable is the string with the column name for filtering the dataframe
    # filterValue is the value to filter the dataframe
    # quantityVariable is the string with the column name for the sum of the quantities
    # countVariable is the string with the column name for the count 
    #titolo is the title of the figure
    
    if len(filterVariable)>0:
        D_temp=D_temp[D_temp[filterVariable]==filterValue]
    D_temp=D_temp.sort_values(date_field)
    D_temp=D_temp.reset_index(drop=True)
    D_temp=D_temp.dropna(subset=[date_field,quantityVariable])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(titolo)
    
    
    
    #plot quantity
    axs[0].plot(D_temp[date_field],D_temp[quantityVariable])
    axs[0].set_title('Quantity trend')
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(45)
    
    #plot lines
    axs[1].plot(D_temp[date_field],D_temp[countVariable])
    axs[1].set_title('Lines trend')
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(45)
    
   
    
    #plt.close('all')
    return fig


# %% PLOT DAILY AND WEEKLY TRENDS WITH SUBPLOTS
    
def plotQuantityTrendWeeklyDaily(D_temp,  date_field='TIMESTAMP_IN', filterVariable=[], filterValue=[],  quantityVariable = 'sum_QUANTITY', countVariable = 'count_TIMESTAMP_IN', titolo=''):
    #the function return a figure with two subplots on for quantities the other for lines
    # D_temp is the input dataframe
    #data_fiels is the string with the column name for the date field
    # filterVariable is the string with the column name for filtering the dataframe
    # filterValue is the value to filter the dataframe
    # quantityVariable is the string with the column name for the sum of the quantities
    # countVariable is the string with the column name for the count 
    #titolo is the title of the figure
    
    if len(filterVariable)>0:
        D_temp=D_temp[D_temp[filterVariable]==filterValue]
    D_temp=D_temp.sort_values(date_field)
    D_temp=D_temp.reset_index(drop=True)
    D_temp=D_temp.dropna(subset=[date_field,quantityVariable])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(titolo)
    
    
    #QUANTITIES
    
    
    # estraggo la serie temporale giornaliera
    timeSeries=pd.DataFrame(D_temp[[date_field,quantityVariable]])
    timeSeries_day=timeSeries.set_index(date_field).resample('D').sum()

    # estraggo la serie temporale settimanale
    timeSeries_week=ts.raggruppaPerSettimana(timeSeries,date_field,quantityVariable,'sum')
    
    # estraggo la serie temporale mensile
    timeSeries_month=ts.raggruppaPerMese(timeSeries,date_field,quantityVariable,'sum')
    
    #plot weekly-daily
    axs[0].plot(timeSeries_day)
    axs[0].plot(timeSeries_week)
    axs[0].plot(timeSeries_month)
    axs[0].set_title('Quantity trend')
    axs[0].legend(['daily time series','weekly time series','monthly time series'])
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(45)
        
    #LINES
    
    
    # estraggo la serie temporale giornaliera
    timeSeries=pd.DataFrame(D_temp[[date_field,countVariable]])
    timeSeries_day=timeSeries.set_index(date_field).resample('D').sum()

    # estraggo la serie temporale settimanale
    timeSeries_week=ts.raggruppaPerSettimana(timeSeries,date_field,countVariable,'sum')
    
    # estraggo la serie temporale settimanale
    timeSeries_month=ts.raggruppaPerMese(timeSeries,date_field,countVariable,'sum')
    
    
    #plot weekly-daily
    axs[1].plot(timeSeries_day)
    axs[1].plot(timeSeries_week)
    axs[1].plot(timeSeries_month)
    axs[1].set_title('Lines trend')
    axs[1].legend(['daily time series','weekly time series','monthly time series'])
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(45)

    
   
    
    #plt.close('all')
    return fig


# %% decompose time series
    #daily decompose

def decomposeTimeSeries(D_time, seriesVariable, samplingInterval='week', date_field='TIMESTAMP_IN',  decompositionModel='additive'):
    #this function defines a graph decomposing a time series
    #D_time is the reference dataframe
    #date_field is the string with the name of the column containing the datetime series
    #seriesVariable is the string with the name of the column containing the series
    #samplingInterval if week it groups the series for week
    #decompositionModel is the argument of seasonal_decompose (additive or multiplicative)

    # estraggo la serie temporale giornaliera
    timeSeries=pd.DataFrame(D_time[[date_field,seriesVariable]])
    timeSeries_analysis=timeSeries.set_index(date_field).resample('D').sum()
    timeSeries_analysis[date_field]=timeSeries_analysis.index.values
    
    
    if samplingInterval=='month':
        timeSeries_analysis=ts.raggruppaPerMese(timeSeries_analysis,date_field,seriesVariable,'sum')
        frequency=min(12, len(timeSeries_analysis)-1) # cerco una frequenza annuale
    elif samplingInterval=='week':
        timeSeries_analysis=ts.raggruppaPerSettimana(timeSeries_analysis,date_field,seriesVariable,'sum')
        frequency= min(4, len(timeSeries_analysis)-1) # cerco una frequenza mensile
    elif samplingInterval=='day':
        timeSeries_analysis=timeSeries_analysis[seriesVariable]
        frequency = min(7, len(timeSeries_analysis)-1) # cerco una frequenza settimanale

    result = seasonal_decompose(timeSeries_analysis, model=decompositionModel, freq=frequency)
    fig=result.plot()
    return fig

# %% detect seasonality using Fourier analysis

def seasonalityWithfourier(D_time, seriesVariable, samplingInterval='week', date_field='TIMESTAMP_IN',titolo=''):
    #this function decompose the seasonal part of a time series using Fourier transform
    #D_time is the reference dataframe
    #seriesVariable is the string with the name of the column containing the series
    #samplingInterval if week it groups the series for week or gay
    #date_field is the string with the name of the column containing the datetime series
    #titolo is the title of the graph
    
    
    # estraggo la serie temporale 
    timeSeries=pd.DataFrame(D_time[[date_field,seriesVariable]])
    timeSeries_analysis=timeSeries.set_index(date_field).resample('D').sum()
    timeSeries_analysis[date_field]=timeSeries_analysis.index.values
    
    if samplingInterval=='month':
        timeSeries_analysis=ts.raggruppaPerMese(timeSeries_analysis,date_field,seriesVariable,'sum')
    elif samplingInterval=='week':
        timeSeries_analysis=ts.raggruppaPerSettimana(timeSeries_analysis,date_field,seriesVariable,'sum')
    elif samplingInterval=='day':
        timeSeries_analysis=timeSeries_analysis[seriesVariable]
    
    y=np.array(timeSeries_analysis)
    D = ts.fourierAnalysis(y)
    
    fig=plt.figure()
    plt.stem(1/D['Frequency_domain_value'], D['Amplitude'])
    plt.title(f"Amplitude spectrum {titolo}")
    plt.xlabel(f"Time domain: {samplingInterval}")
    plt.ylabel('Amplitude')
    return fig
