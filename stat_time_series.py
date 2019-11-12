# -*- coding: utf-8 -*-
#import py_compile
#py_compile.compile('ZO_ML_timeSeries.py')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import itertools
import warnings
from numpy import fft


def timeStampToDays(series):
    #converto da una pandas timestamp a un numero in giorni
    D=series.dt.components['days']
    H=series.dt.components['hours']
    M=series.dt.components['minutes']
    result=D + H/24 + M/(60*24)
    return result


def raggruppaPerSettimana(df,variable,tipo):
    df['DatePeriod'] = pd.to_datetime(df['DatePeriod']) - pd.to_timedelta(7, unit='d')
    if tipo=='count':
        df = df.groupby([pd.Grouper(key='DatePeriod', freq='W-MON')])[variable].size()
    elif tipo=='sum':
         df = df.groupby([pd.Grouper(key='DatePeriod', freq='W-MON')])[variable].sum()
    df=df.sort_index()
    return df

def raggruppaPerGiornoDellaSettimana(df,variable):
    cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Weekday']=df.DatePeriod.dt.weekday_name
    df['Weekday'] = df['Weekday'].astype('category', categories=cats, ordered=True)
    D_grouped=df.groupby(['Weekday']).agg({variable:['size','mean','std']})
    D_grouped.columns = D_grouped.columns.droplevel(0)
    D_grouped['mean']=np.round(D_grouped['mean'],2)
    D_grouped['std']=np.round(D_grouped['std'],2)
    return D_grouped

def assegnaGiornoSettimana(df,dateperiodColumn):
    dayOfTheWeek=df[dateperiodColumn].dt.weekday_name
    weekend=(dayOfTheWeek=='Sunday') | (dayOfTheWeek=='Saturday')
    weekEnd=weekend.copy()
    weekEnd[weekend]='Weekend'
    weekEnd[~weekend]='Weekday'
    return dayOfTheWeek,weekEnd
   

# In[1]: autofit
def autoSARIMAXfit(y,minRangepdq, maxRangepdqy,seasonality):
    minRangepdq=np.int(minRangepdq)
    maxRangepdqy=np.int(maxRangepdqy)
    seasonality=np.int(seasonality)
    
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(minRangepdq, maxRangepdqy)
    
# Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    incumbentError=9999999999;
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
                if(results.aic<incumbentError):
                    bestModel=mod
                    incumbentError=results.aic
                
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    return bestModel

# In[1]: ACF and PACF
def ACF_PACF_plot(series):
    
    
    fig=plt.figure()
    plt.subplot(131) 
    plt.plot(series,'skyblue')
    plt.xticks(rotation=30)
    plt.title('Time Series')
    
    
    
    lag_acf = acf(series, nlags = 20)
    lag_pacf = pacf(series, nlags = 20)

    plt.subplot(132) 
    plt.stem(lag_acf,linefmt='skyblue',markerfmt='d')
    plt.axhline(y=0,linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='r')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='r')
    plt.title('ACF')
    plt.xlabel('time lag')
    plt.xlabel('ACF value')


    plt.subplot(133) 
    plt.stem(lag_pacf,linefmt='skyblue',markerfmt='d')
    plt.axhline(y=0,linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='r')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='r')
    plt.title('PACF')
    plt.xlabel('time lag')
    plt.xlabel('PACF value')
    return fig

def detrendByRollingMean(series,seasonalityPeriod):
    rolling_mean = series.rolling(window = seasonalityPeriod).mean()
    detrended = series.Series - rolling_mean
    return detrended


def ARIMAfit(series,p,d,q):
    series=series[~np.isnan(series)]
    model = ARIMA(series, order=(p, d, q))  
    results_AR = model.fit(disp=-1)  
    plt.plot(series)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('ARIMA fit p='+str(p)+' q='+str(q)+' d='+str(d))
    
    plt.figure()
    results_AR.plot_diagnostics(figsize=(15, 12))
    return 1
   

def forecastSARIMAX(addSeries,minRangepdq, maxRangepdqy,seasonality, NofSteps,titolo):
    NofSteps=np.int(NofSteps)
    #residui=plt.figure()
    result=autoSARIMAXfit(addSeries,minRangepdq, maxRangepdqy,seasonality)
    results=result.fit()
    residui=results.plot_diagnostics(figsize=(15, 12))
    
    
    forecast=plt.figure()
    pred = results.get_prediction(start=len(addSeries)-1,end=len(addSeries)+NofSteps, dynamic=True)
    pred_ci = pred.conf_int()
    
    ax = addSeries.plot(label='observed',color='orange')
    pred.predicted_mean.plot(ax=ax, label='Dynamic forecast', color='r', style='--', alpha=.7)
    
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='y', alpha=.2)
    
    ax.set_xlabel('Timeline')
    ax.set_ylabel('Series value')
    plt.title('Forecast: '+titolo)
    plt.legend()
    return residui,forecast

def fourierAnalysis(y):
    
    y=y.reshape(len(y),)
    N = len(y) #600 campioni
    T=1 #un campione alla settimana
    
    #plt.figure()
    t = np.arange(0, len(y)).reshape(len(y),)
    p = np.polyfit(t, y, 1)         # find linear trend in x
    y_notrend = y - p[0] * t 
    #plt.plot(x,y_notrend)
    #plt.title('detrended signal')
    #plt.xlabel('settimane')
    
    #calcolo fourier transform
    yf = np.fft.fft(y_notrend)
    
    #filtro i valori più significativi (solo le frequenze il cui picco spiega almeo il 10% della stagionalità)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    amplitude=2.0/N * np.abs(yf[0:N//2])
    weeks=1/xf
    
    
    #plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
    #plt.grid()
    #plt.show()
    
    data={'Frequenza':xf,'Settimana':weeks,'Ampiezza':amplitude  }
    D=pd.DataFrame(data)
    D=D.replace([np.inf, -np.inf], np.nan)
    D=D.dropna()
    D=D.sort_values(['Ampiezza'],ascending=False)
    D['perc']=D.Ampiezza/np.sum(D.Ampiezza)
    D=D[D['perc']>0.1]
    D['Settimana']=np.round(D.Settimana,0)
    return D.Settimana
    
    
    





