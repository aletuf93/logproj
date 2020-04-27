# -*- coding: utf-8 -*-
import datetime as date
import numpy as np
import pandas as pd


# import stat packages
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error

#import graph packages
import matplotlib.pyplot as plt
import plotly.offline as py

#import industrial packages
import logproj.stat_time_series as ts

# import utiliy functions
from dashboard.Zutilities import creaCartella

# %% PREDICTION FBPROPHET
    
def predictWithFBPROPHET(D_series, timeVariable, seriesVariable, prediction_results, titolo, samplingInterval='week', predictionsLength=52):
    #D_time is a dataframe containing the timeseries and the values
    #timeVariable is a string with the name of the column of the dataframe containing timestamps
    #seriesVariable is a string with the name of the column of the dataframe containing values
    #predictionsLength is an int with the number of periods to predict
    #prediction_results is the path where to save the output
    #samplingInterval if week it groups the series for week
    # titolo is the title to save the output figure
    
    
    # estraggo la serie temporale 
    timeSeries=pd.DataFrame(D_series[[timeVariable,seriesVariable]])
    timeSeries_analysis=timeSeries.set_index(timeVariable).resample('D').sum()
    timeSeries_analysis[timeVariable]=timeSeries_analysis.index.values
    
    if samplingInterval=='month':
        timeSeries_analysis=ts.raggruppaPerMese(timeSeries_analysis,timeVariable,seriesVariable,'sum')
    elif samplingInterval=='week':
        timeSeries_analysis=ts.raggruppaPerSettimana(timeSeries_analysis,timeVariable,seriesVariable,'sum')
    elif samplingInterval=='day':
        timeSeries_analysis=timeSeries_analysis[seriesVariable]


    #prepare input dataframe
    timeSeries_analysis=pd.DataFrame([timeSeries_analysis.index.values, timeSeries_analysis]).transpose()
    timeSeries_analysis.columns=['ds','y']

    m = Prophet()
    m.fit(timeSeries_analysis)
    
    #make predictions
    future = m.make_future_dataframe(periods=predictionsLength)
    #future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    
    
    #evaluate model goodness
    MSE = mean_squared_error(timeSeries_analysis.y, forecast.yhat[0:len(timeSeries_analysis.y)])

    # Output figure in matplotlib
    forecast_fig = m.plot(forecast)
    components_fig = m.plot_components(forecast)


    
    
    #Output with plotly
    #py.init_notebook_mode()
    fig = plot_plotly(m, forecast)  # This returns a plotly Figure
    py.iplot(fig)
    py.plot(fig, filename = f"{prediction_results}\\prophet_{titolo}.html", auto_open=False)
    return m, forecast_fig, components_fig, MSE



# %% PREDICTIONS ARIMA

def predictWithARIMA(D_series, seriesVariable, samplingInterval='week', date_field='TIMESTAMP_IN',titolo='',signifAlpha=0.05,maxValuesSelected=2):
    
    #this function applies predictions using ARIMA models
    
    #D_series is the reference dataframe
    #date_field is the string with the name of the column containing the datetime series
    #seriesVariable is the string with the name of the column containing the series
    #samplingInterval if week it groups the series for week
    #signifAlpha is the significance level (0.1 , 0.05, 0.01) to accept or reject the null hypothesis of Dickey fuller
    #maxValuesSelected int defining the number of significant lags to consider in ACF and PACF
    
    #the function returns 
    # fig_CF with the PACF and ACF figure
    #figure_forecast the forecast figure, 
    #figure_residuals the residual figure, 
    #resultModel the model resulting parameters
    
    
    
    
    # estraggo la serie temporale 
    timeSeries=pd.DataFrame(D_series[[date_field,seriesVariable]])
    timeSeries_analysis=timeSeries.set_index(date_field).resample('D').sum()
    timeSeries_analysis[date_field]=timeSeries_analysis.index.values
    
    if samplingInterval=='month':
        timeSeries_analysis=ts.raggruppaPerMese(timeSeries_analysis,date_field,seriesVariable,'sum')
    elif samplingInterval=='week':
        timeSeries_analysis=ts.raggruppaPerSettimana(timeSeries_analysis,date_field,seriesVariable,'sum')
    elif samplingInterval=='day':
        timeSeries_analysis=timeSeries_analysis[seriesVariable]
            
            
    #transform series to stationarity
    seriesVariable='count_TIMESTAMP_IN'
    stationary_series, stationary_model = ts.transformSeriesToStationary(timeSeries_analysis,signifAlpha=signifAlpha)
    
    #aggiungere l'uscita del modello stazionario e il return
    
    #se sono riuscito a frasformare la serie in stazionaria proseguo
    if len(stationary_series)>1:
        
        #detect ACF and PACF
        fig_CF, D_acf_significant, D_pacf_significant = ts.ACF_PACF_plot(stationary_series)
        params = ts.returnsignificantLags(D_pacf_significant, D_acf_significant, maxValuesSelected)
        
        # Running ARIMA fit, consider that
        
        
        
        figure_forecast, figure_residuals, resultModel = ts.SARIMAXfit(stationary_series, params)
        
        return stationary_model, fig_CF, figure_forecast, figure_residuals, resultModel
    else: #cannot make the series stationary, cannot use ARIMA
        return [], [], [], [], []
