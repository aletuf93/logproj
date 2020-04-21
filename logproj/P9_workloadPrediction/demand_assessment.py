# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logproj.stat_time_series as ts
from statsmodels.tsa.seasonal import seasonal_decompose
from logproj.P8_performanceAssessment.utilities_movements import getCoverageStats



# %%
def getAdvanceInPlanning(D_mov,loadingptafield='LOADING_TIME_WINDOWS_PROVISIONAL_START'):
    #la funzione calcola la distribuzione del tempo di anticipo di pianificazione
    #come differenza fra il timestamp_ in e la finestra temporale di carico

    output_figure={}
    output_data={}
    output_coverage={}

    #filterNan
    D_mov_filtered = D_mov[['TIMESTAMP_IN',loadingptafield]].dropna()

    if len(D_mov_filtered)==0:
        return output_figure, pd.DataFrame(['No PTA fields to perform this analysis'])
    if loadingptafield=='TIMESTAMP_IN': #se uso la stessa colonna ottengo 0
        mean_advanceInPlanning =  std_advanceInPlanning =0
        advanceInPlanningDistribution =[]
    else:
        advanceInPlanning=D_mov_filtered[loadingptafield]-D_mov_filtered['TIMESTAMP_IN']
        advanceInPlanningD=advanceInPlanning.dt.components['days']
        advanceInPlanningH=advanceInPlanning.dt.components['hours']
        advanceInPlanningM=advanceInPlanning.dt.components['minutes']
        advanceInPlanning=advanceInPlanningD + advanceInPlanningH/24 +advanceInPlanningM/(60*24)
        advanceInPlanning=advanceInPlanning[advanceInPlanning>0]
        mean_advanceInPlanning=np.mean(advanceInPlanning)
        std_advanceInPlanning=np.std(advanceInPlanning)
        advanceInPlanningDistribution=advanceInPlanning



    if len(advanceInPlanningDistribution)>0:
        #Advance in planning
        fig_planningAdvance=plt.figure()
        plt.title('Days of advance in booking')
        plt.hist(advanceInPlanning,color='orange',bins=np.arange(0,max(advanceInPlanningDistribution),1))
        plt.xlabel('days')
        plt.ylabel('N.ofBooks')

        #output_figure
        output_figure['ADVANCE_IN_PLANNING']=fig_planningAdvance

    #output_data
    output_data['ADVANCE_PLANNING_MEAN']= mean_advanceInPlanning
    output_data['ADVANCE_PLANNING_STD']=std_advanceInPlanning
    output_data['SERIES']=advanceInPlanningDistribution

    #get coverage
    output_coverage['ADVANCE_PLANNING_MEAN'] = getCoverageStats(D_mov,loadingptafield,capacityField='QUANTITY')
    output_coverage['ADVANCE_PLANNING_STD']=output_coverage['ADVANCE_PLANNING_MEAN']
    output_coverage['SERIES']=output_coverage['ADVANCE_PLANNING_MEAN']

    D_global=pd.DataFrame([output_data,output_coverage]).transpose()
    D_global.columns=['VALUE','ACCURACY']


    return output_figure, D_global

# %%
def bookingStatistics(D_mov,capacityField='QUANTITY',
                      timeVariable='TIMESTAMP_IN',
                      samplingInterval=['day','week','month']):
    #Analisi trend mensili, settimanali, giornalieri e per giorno della settimana
    #timeVariable e' una variabile di raggruppamento base tempo
    #capacityField e' la variabile di capacita' per studiare le coperture


    #creo dizionari di risultati
    imageResults={}
    dataframeResults={}
    dataResults_trend={}
    coverage_stats={}

    #calcolo le coperture
    accuracy, _ = getCoverageStats(D_mov,analysisFieldList=timeVariable,capacityField=capacityField)

    D_OrderTrend=D_mov.groupby([timeVariable]).size().reset_index()
    D_OrderTrend.columns=['DatePeriod','Orders']
    D_OrderTrend=D_OrderTrend.sort_values(['DatePeriod'])
    #D_OrderTrend['DatePeriod']=pd.to_datetime(D_OrderTrend['DatePeriod'])


    for spInterval in samplingInterval:
        if  spInterval == 'month':
            timeSeries_analysis=ts.raggruppaPerMese(D_OrderTrend,'DatePeriod','Orders','sum')

        elif spInterval == 'week' :
            timeSeries_analysis=ts.raggruppaPerSettimana(D_OrderTrend,'DatePeriod','Orders','sum')

        elif spInterval == 'day':
            timeSeries_analysis=D_OrderTrend.set_index('DatePeriod')
            timeSeries_analysis=timeSeries_analysis['Orders']


        #trend giornaliero
        fig1=plt.figure()
        plt.plot(timeSeries_analysis.index.values,timeSeries_analysis,color='orange')
        plt.title(f"TREND: {timeVariable} per {spInterval}")
        plt.xticks(rotation=30)
        imageResults[f"trend_{spInterval}"]=fig1

        #distribuzione
        fig2=plt.figure()
        plt.hist(timeSeries_analysis,color='orange')
        plt.title(f"Frequency analysis of {timeVariable} per {spInterval}")
        plt.xlabel(f"{timeVariable}")
        plt.ylabel(f"{spInterval}")
        imageResults[f"pdf_{spInterval}"]=fig2
        #fig1.savefig(dirResults+'\\02-ContainerPDFDaily.png')

        daily_mean=np.mean(timeSeries_analysis)
        daily_std=np.std(timeSeries_analysis)

        #calcolo i valori
        dataResults_trend[f"{timeVariable}_{spInterval}_MEAN"] = daily_mean
        dataResults_trend[f"{timeVariable}_{spInterval}_STD"] = daily_std

        #assegno le coperture
        coverage_stats[f"{timeVariable}_{spInterval}_MEAN"] = accuracy
        coverage_stats[f"{timeVariable}_{spInterval}_STD"] = accuracy


    #salvo dataframe con i risultati dei trend e le coperture
    D_trend_stat=pd.DataFrame([dataResults_trend,coverage_stats]).transpose()
    D_trend_stat.columns=['VALUE','ACCURACY']
    dataframeResults['trend_df']=D_trend_stat

    #distribuzione per giorno della settimana
    D_grouped=ts.raggruppaPerGiornoDellaSettimana(D_OrderTrend,timeVariable='DatePeriod',seriesVariable ='Orders')
    D_grouped['accuracy']=[accuracy for i in range(0,len(D_grouped))]
    dataframeResults['weekday_df']=D_grouped
    #D_grouped.to_excel(dirResults+'\\02-ContainerWeekday.xlsx')

    fig3=plt.figure()
    plt.bar(D_grouped.index.values,D_grouped['mean'],color='orange')
    plt.title(f"N.of {timeVariable} per day of the week")
    plt.xlabel('day of the week')
    plt.ylabel('Frequency')
    imageResults[f"pdf_dayOfTheWeek"]=fig3
    #fig1.savefig(dirResults+'\\02-ContainerPerweekDay.png')


    #D_movDaily.to_excel(dirResults+'\\02-ContainerDailyStats.xlsx')
    return imageResults, dataframeResults

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

    if len(timeSeries_analysis)<2*frequency:
        print(f"Not enough values to decompose series with sampling interval {samplingInterval}")
        return plt.figure()
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
