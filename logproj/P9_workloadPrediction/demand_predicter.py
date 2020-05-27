# -*- coding: utf-8 -*-
#import datetime as date
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as date
import numpy as np


# import stat packages
from fbprophet import Prophet
#from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error

#import graph packages
#import matplotlib.pyplot as plt
import plotly.offline as py

#import industrial packages
import logproj.stat_time_series as ts
from logproj.utilities import creaCartella


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
    
# %%
def LOOP_PREDICT_SARIMA(D_time, date_field, qtyVariable, countVariable,prediction_results_path,filterVariable=[], samplingInterval='week'):
    
    def nestedPredictionSARIMA(D_results, ss, nomecartella):
        #create folder with results
        _, current_dir_results = creaCartella(prediction_results_path,f"{str(nomecartella)}")
            
        print(f"***********{ss}*************")
        if len(ss)>0:
            D_series=D_time[D_time[filterVariable]==ss]
        else:
            D_series=D_time
        
        #save report txt
        file = open(f"{current_dir_results}\\{reportFilename}.txt", "w") 
        file.write(f"{str(date.datetime.now())} STARTING PREDICTIONS \r") 
        file.close() 
        
        # set initial
        tot_qty = np.nansum(D_series[qtyVariable])
        tot_lns = np.nansum(D_series[countVariable])
        QTY_STATIONARY_TRANSFORM=''
        QTY_SARIMA_MODEL=''
        QTY_SARIMA_ACCURACY = ''
        LNS_STATIONARY_TRANSFORM=''
        LNS_SARIMA_MODEL=''
        LNS_SARIMA_ACCURACY = ''
        
        if len(D_series)>=12: #I need at least 12 points (e.g. 1 year expressed in  months)
            
            # predict quantities
            stationary_model, fig_CF, figure_forecast, figure_residuals, resultModel = predictWithARIMA(D_series, 
                                                                                                        seriesVariable=qtyVariable, 
                                                                                                        samplingInterval=samplingInterval, 
                                                                                                        date_field=date_field,
                                                                                                        signifAlpha=0.05,
                                                                                                        maxValuesSelected=2)
            
            if len(stationary_model)>0:
                #save figures
                fig_CF.get_figure().savefig(f"{current_dir_results}\\{ss}_quantities_CF.png")
                figure_forecast.savefig(f"{current_dir_results}\\{ss}_quantities_ARIMA_forecast.png")
                figure_residuals.savefig(f"{current_dir_results}\\{ss}_quantities_ARIMA_residuals.png")
                plt.close('all')
            
                #save params
                QTY_STATIONARY_TRANSFORM=stationary_model
                QTY_SARIMA_MODEL=str({'p':resultModel['p'],'d':resultModel['d'],'q':resultModel['q']})
                QTY_SARIMA_ACCURACY = resultModel['aic']
                
                #save report txt
                file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
                file.write(f"{str(date.datetime.now())} quantities: Predictions built\r") 
                file.close() 
            else:
                #save report txt
                file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
                file.write(f"{str(date.datetime.now())} quantities: no stationary series, no ARIMA Predictions built\r") 
                file.close() 
                
            
            #predict lines
            stationary_model, fig_CF, figure_forecast, figure_residuals, resultModel = predictWithARIMA(D_series, 
                                                                                                        seriesVariable=countVariable, 
                                                                                                        samplingInterval=samplingInterval, 
                                                                                                        date_field=date_field,
                                                                                                        signifAlpha=0.05,
                                                                                                        maxValuesSelected=2)
            
            if len(stationary_model) >0: # se le predizioni sono riuscite
                #save figures
                fig_CF.get_figure().savefig(f"{current_dir_results}\\{ss}_lines_CF.png")
                figure_forecast.savefig(f"{current_dir_results}\\{ss}_lines_ARIMA_forecast.png")
                figure_residuals.savefig(f"{current_dir_results}\\{ss}_lines_ARIMA_residuals.png")
                plt.close('all')
            
                LNS_STATIONARY_TRANSFORM=stationary_model
                LNS_SARIMA_MODEL=str({'p':resultModel['p'],'d':resultModel['d'],'q':resultModel['q']})
                LNS_SARIMA_ACCURACY = resultModel['aic']
            
            
                #save report txt
                file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
                file.write(f"{str(date.datetime.now())} lines: Predictions built\r") 
                file.close() 
            else:
                #save report txt
                file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
                file.write(f"{str(date.datetime.now())} lines: no stationary series, no ARIMA Predictions built\r") 
                file.close() 
            
            
        
        
        else:
            
            
            #save report txt
            file = open(f"{current_dir_results}\\{reportFilename}.txt", "w") 
            file.write(f"{str(date.datetime.now())} Not ehough input points to build a time series\r") 
            file.close() 
            
            
        # append results to dataframe
        D_results=D_results.append(pd.DataFrame([[ss, tot_qty, tot_lns, QTY_STATIONARY_TRANSFORM, QTY_SARIMA_MODEL,
                                                QTY_SARIMA_ACCURACY, LNS_STATIONARY_TRANSFORM, LNS_SARIMA_MODEL,
                                                LNS_SARIMA_ACCURACY]], columns = D_results.columns))
        return D_results
    
    
    
    
    
    reportFilename='report_SARIMA'
    
    resultscolumn=['SERVICETYPE','QUANTITIES','LINES','QTY_STATIONARY_TRANSFORM','QTY_SARIMA_MODEL','QTY_SARIMA_ACCURACY',
               'LNS_STATIONARY_TRANSFORM','LNS_SARIMA_MODEL','LNS_SARIMA_ACCURACY']
    
    D_results=pd.DataFrame(columns=resultscolumn)
    print("CIAOOOO")
    print(D_results)
    # genero i trend globali
    D_results = nestedPredictionSARIMA(D_results,[], nomecartella='globalResults')
    
    if len(filterVariable)>0:
        #itero sulle famiglie di prodotto
        st = list(set(D_time[filterVariable]))
        
        for ss in st:
            #ss='zz'
            try:
                D_results = nestedPredictionSARIMA(D_results, ss, nomecartella=ss)
            except Exception as e:
                print(f"*=*=*=*=*=*=ERROR*=*=*= {e}")
        #ss='zz'
        
    
            
            
            
    #  SAVE dataframe results
    D_results.to_excel(f"{prediction_results_path}\\pred_results_SARIMA.xlsx")
    return True


# %%

#PROD_PREDICT_SARIMA(D_time, qtyVariable, countVariable,prediction_results_path,filterVariable=[], samplingInterval='week'):
def LOOP_PREDICT_FBPROPHET(D_time, timeVariable, qtyVariable, countVariable, prediction_results_path, filterVariable=[], samplingInterval='week' ):
    
    
    def nestedPredictionFBprophet(D_results, ss, nomecartella):
        #create folder with results
        _, current_dir_results = creaCartella(prediction_results_path,f"{str(nomecartella)}")
            
        print(f"***********{ss}*************")
        if len(ss)>0:
            D_series=D_time[D_time[filterVariable]==ss]
        else:
            D_series=D_time
        
        #save report txt
        file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
        file.write(f"{str(date.datetime.now())} STARTING PREDICTIONS \r") 
        file.close() 
        
        # set initial
        tot_qty = np.nansum(D_series[qtyVariable])
        tot_lns = np.nansum(D_series[countVariable])
        MSE_QTY = ''
        MSE_LNS = ''
        
        
        if len(D_series)>=12: #I need at least 12 points (e.g. 1 year expressed in  months)
            
            # predict quantities
            m, forecast_fig, components_fig, MSE_result = predictWithFBPROPHET(D_series, 
                                                                               timeVariable, 
                                                                               qtyVariable, 
                                                                               current_dir_results, 
                                                                               samplingInterval=samplingInterval, predictionsLength=52, titolo='qty')
            
            
            
            forecast_fig.savefig(f"{current_dir_results}\\{ss}_quantities_FBPROPHET_forecast.png")
            components_fig.savefig(f"{current_dir_results}\\{ss}_quantities_FBPROPHET_comp.png")
            plt.close('all')
            
            #save params
            MSE_QTY=MSE_result
            
                
            #save report txt
            file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
            file.write(f"{str(date.datetime.now())} quantities: Predictions built\r") 
            file.close() 
            
       
                
            
            # predict quantities
            m, forecast_fig, components_fig, MSE_result = predictWithFBPROPHET(D_series, timeVariable, countVariable, current_dir_results, samplingInterval=samplingInterval, predictionsLength=52, titolo='lines')
            
            
            
            forecast_fig.savefig(f"{current_dir_results}\\{ss}_quantities_FBPROPHET_forecast.png")
            components_fig.savefig(f"{current_dir_results}\\{ss}_quantities_FBPROPHET_comp.png")
            plt.close('all')
            
            #save params
            MSE_LNS=MSE_result
            
                
            #save report txt
            file = open(f"{current_dir_results}\\{reportFilename}.txt", "a") 
            file.write(f"{str(date.datetime.now())} quantities: Predictions built\r") 
            file.close() 
            
        
            
            
        
        
        else:
            
            
            #save report txt
            file = open(f"{current_dir_results}\\{reportFilename}.txt", "w") 
            file.write(f"{str(date.datetime.now())} Not ehough input points to build a time series\r") 
            file.close() 
            
            
        # append results to dataframe
        D_results=D_results.append(pd.DataFrame([[ss, tot_qty, tot_lns, MSE_QTY, MSE_LNS]], columns = D_results.columns))
        return D_results
    
    reportFilename='report_fbProphet'
    resultscolumn=['SERVICETYPE','QUANTITIES','LINES','MSE_QTY','MSE_LNS']
    D_results=pd.DataFrame(columns=resultscolumn)
    
    
    # genero i trend globali
    D_results = nestedPredictionFBprophet(D_results,[], nomecartella='globalResults')
    
    if len(filterVariable)>0:
        #itero sulle famiglie di prodotto
        st = list(set(D_time[filterVariable]))
        
        for ss in st:
            #ss='zz'
            try:
                D_results = nestedPredictionFBprophet(D_results, ss, nomecartella=ss)
            except Exception as e:
                print(f"*=*=*=*=*=*=ERROR*=*=*= {e}")
        
        
            
            
            
    #  SAVE dataframe results
    D_results.to_excel(f"{prediction_results_path}\\pred_results_FBPROPHET.xlsx")
    return True
