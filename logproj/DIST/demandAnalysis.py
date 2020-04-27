# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logproj.stat_time_series as ts

from logproj.DIST.globalBookingAnalysis import getCoverageStats
from logproj.DIST.voyageAnalysis import createTabellaMovimenti

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

# %% schedule statistics
    
def checkPlannedActual(D_mov,locfrom = 'LOADING_NODE',                     
                            locto= 'DISCHARGING_NODE',
                            capacityField='QUANTITY',
                            voyagefield ='VOYAGE_CODE',
                            vehiclefield='VEHICLE_CODE',
                            timeColumns={}):
    
    df_results={}
    output_figure={}
    
    D = createTabellaMovimenti( D_mov,
                                locfrom = locfrom,                     
                                locto= locto,
                                capacityField=capacityField,
                                timeColumns=timeColumns
                                )
    if any(column not in D.columns for column in ['PTA','PTD','ATA','ATD']):
        print ("WARNING: no actual and provisional columns in D_mov")
        return output_figure, df_results
    accuracy, _ = getCoverageStats(D_mov,analysisFieldList=[locfrom, locto, voyagefield, vehiclefield,*list(timeColumns.values())
                                                                          ],capacityField='QUANTITY')
    
    D_movimenti=D.groupby([vehiclefield,'Location','PTA','PTD','ATA','ATD',voyagefield])['Movementquantity'].sum().reset_index()
    D_movimenti['AsPlanned']=True #memorizzo anche in tabella movimenti se ho rispettato le route
    colsCheckRoute=['VoyageCode','PlanPerformed']
    results_route=pd.DataFrame(columns=colsCheckRoute)
    
    colsCheckArcs=['VoyageCode','plannedLocation','actualLocation']
    results_arcExchange=pd.DataFrame(columns=colsCheckArcs)
    
    #identifico le route
    routeCode=np.unique(D_movimenti[voyagefield][~D_movimenti[voyagefield].isna()])
    for i in range(0,len(routeCode)):
        codiceRoute=routeCode[i]
        dataRoute=D_movimenti[D_movimenti[voyagefield]==codiceRoute]
        
        #ordino per PLANNED
        sortpl=dataRoute.sort_values(by='PTA')
        ordinePlanned=sortpl.index.values
        
        #ordino per ACTUAL
        sortact=dataRoute.sort_values(by='ATA')
        ordineActual=sortact.index.values
        
        check=all(ordineActual==ordinePlanned)
        
        if(check): #la route è eseguita come pianificato
            #aggiorno tabella voyage
            temp=pd.DataFrame([[codiceRoute,True]],columns=colsCheckRoute);
            results_route=results_route.append(temp)
        else: #la route non è eseguita come pianificato
            #aggiorno tabella voyage
            temp=pd.DataFrame([[codiceRoute,False]],columns=colsCheckRoute);
            results_route=results_route.append(temp)
            
            #aggiorno tabella  arc exchange
            
            #identifico gli indici incriminati
            indexFrom=sortpl[~(ordineActual==ordinePlanned)].index.values
            indexTo=sortact[~(ordineActual==ordinePlanned)].index.values
            
            locFrom=dataRoute.Location[indexFrom]
            locTo=dataRoute.Location[indexTo]
            for j in range(0,len(locFrom)):
                temp=pd.DataFrame([[codiceRoute,locFrom.iloc[j],locTo.iloc[j]]],columns=colsCheckArcs);
                results_arcExchange=results_arcExchange.append(temp)
            
            #Segno in tabella movimenti se il tragitto pianificato è stato rispettato
            D_movimenti.loc[(D_movimenti[voyagefield]==codiceRoute) & (D_movimenti.Location.isin(locFrom)),'AsPlanned']=False
                
    
            
    #calcolo statistiche sulle modifiche
    stat_exchange=results_arcExchange.groupby(['plannedLocation','actualLocation']).size().reset_index()
    stat_exchange.rename(columns={0:'count'}, inplace=True)
    stat_exchange=stat_exchange.sort_values(by='count',ascending=False)
    
    stat_exchange['accuracy']= [accuracy for i in range(0,len(stat_exchange))]
    results_route['accuracy']= [accuracy for i in range(0,len(results_route))]
    
    df_results['routeExchange'] = stat_exchange
    df_results['routeExecutedAsPlanned'] = results_route
    
    
    
    #creo pie-chart con la percentuale di route rispettate
    
    
    sizes=results_route.groupby(['PlanPerformed']).size()
    labels=sizes.index.values
    explode = 0.1*np.ones(len(sizes))
    
    fig1, ax1 = plt.subplots(figsize=(20,10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Route as planned')
    output_figure['routeAsPlannedPie']=fig1
    
    
    
    # calcolo un differenziale planned-actual anche a seconda di quanto siamo lontani nel tempo dalla creazione del Record
    
    D_movimenti['latenessTD']=lateness_TD=ts.timeStampToDays(D_movimenti.ATD-D_movimenti.PTD)
    D_movimenti['tardinessTD']=tardiness_TD=lateness_TD.clip(0,None) #azzera tutti i valori fuori dall'intervallo 0, +inf
    lateness_TD_mean=np.mean(lateness_TD)
    tardiness_TD_mean=np.mean(tardiness_TD)
    
    lateness_TA=ts.timeStampToDays(D_movimenti.ATA-D_movimenti.PTA)
    tardiness_TA=lateness_TA.clip(0,None)
    lateness_TA_mean=np.mean(lateness_TA)
    tardiness_TA_mean=np.mean(tardiness_TA)
    
    
    gap_handling=ts.timeStampToDays((D_movimenti.ATD-D_movimenti.ATA) - (D_movimenti.PTD-D_movimenti.PTA))
    handling_gap_mean=np.mean(gap_handling)
    
    cols=['mean lateness - dep.','mean lateness - arr.','mean tardiness - dep.','mean tardiness - arr.','mean handling gap']
    schedule_results=pd.DataFrame([[lateness_TD_mean,lateness_TA_mean,tardiness_TD_mean,tardiness_TA_mean,handling_gap_mean]],columns=cols)
    schedule_results['accuracy']= [accuracy for i in range(0,len(schedule_results))]
    
    df_results['schedule_results'] = schedule_results
    
    return output_figure, df_results