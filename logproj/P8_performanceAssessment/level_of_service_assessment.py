
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logproj import stat_time_series as ts
from logproj.P8_performanceAssessment.utilities_movements import getCoverageStats
from logproj.P8_performanceAssessment.vehicle_assessment import createTabellaMovimenti
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

# %%
#calculate the average time spent by products on a vehicle
def travelTimedistribution(D_mov,
                           capacityField='QUANTITY',
                           loadingTA='PTA_FROM',
                           loadingTD='PTD_FROM',
                           dischargingTA='PTA_TO',
                           dischargingTD='PTD_TO',
                           ):

    df_traveltime=pd.DataFrame(columns=['U_L_BOUND','TIME_MEAN','TIME_STD'])
    imageResults={}

    #get coverage
    accuracy_ub, _ = getCoverageStats(D_mov,analysisFieldList=[dischargingTD, loadingTA],capacityField=capacityField)


    #Expected travel time per container (UPPER BOUND)
    ExpectedTravelTime_ub=ts.timeStampToDays(D_mov[dischargingTD]-D_mov[loadingTA])


    ExpectedTravelTime_ub=ExpectedTravelTime_ub[ExpectedTravelTime_ub>0]
    mean_ExpectedTravelTime=np.mean(ExpectedTravelTime_ub)
    std_ExpectedTravelTime=np.std(ExpectedTravelTime_ub)

    data={'U_L_BOUND':'upperBound',
          'TIME_MEAN':mean_ExpectedTravelTime,
          'TIME_STD':std_ExpectedTravelTime,
          'accuracy':str(accuracy_ub) }
    temp=pd.DataFrame(data,index=[0])
    df_traveltime=df_traveltime.append(temp)



    #aspetto di graficare il LB e poi salvo


    #get coverage
    accuracy_lb, _ = getCoverageStats(D_mov,analysisFieldList=[dischargingTA, loadingTD],capacityField=capacityField)
    #Expected travel time per container (LOWER BOUND)
    ExpectedTravelTime_lb=ts.timeStampToDays(D_mov[dischargingTA]-D_mov[loadingTD])


    ExpectedTravelTime_lb=ExpectedTravelTime_lb[ExpectedTravelTime_lb>0]
    mean_ExpectedTravelTime=np.mean(ExpectedTravelTime_lb)
    std_ExpectedTravelTime=np.std(ExpectedTravelTime_lb)




    data={'U_L_BOUND':'lowerBound',
          'TIME_MEAN':mean_ExpectedTravelTime,
          'TIME_STD':std_ExpectedTravelTime,
          'accuracy':str(accuracy_lb)}
    temp=pd.DataFrame(data,index=[0])
    df_traveltime=df_traveltime.append(temp)

    # salvo figura
    #definisco udm
    udm='days'
    value_ub=ExpectedTravelTime_ub
    value_lb=ExpectedTravelTime_lb
    if mean_ExpectedTravelTime<1/24/60:
        udm='minutes'
        value_ub=ExpectedTravelTime_ub*24*60
        value_lb=ExpectedTravelTime_lb*24*60

    elif mean_ExpectedTravelTime<1: #se ho dei numeri inferiori all'unita', cambio udm
        udm='hours'
        value_ub=ExpectedTravelTime_ub*24
        value_lb=ExpectedTravelTime_lb*24

    fig1=plt.figure()
    plt.hist(value_ub,color='orange')
    plt.hist(value_lb,color='blue',alpha=0.6)
    plt.title(f"Travel time ({udm})")
    plt.xlabel(f"{udm}")
    plt.ylabel('Quantity')
    plt.legend(['Upper bound','Lower bound'])

    imageResults[f"travel_time_per_movement"]=fig1

    return imageResults, df_traveltime

# %% level of service
#define the level of service
def calculateLoS(D_mov,
                           capacityField='QUANTITY',
                           timeColumns={}
                           ):

    output_figure={}
    coverages=pd.DataFrame()


    if all( column in timeColumns.keys() for column in ['loadingptd','dischargingpta',
                                                        'loadingatd','dischargingata']):
        columnsNeeded = [timeColumns['loadingptd'], timeColumns['dischargingpta'],
                         timeColumns['loadingatd'], timeColumns['dischargingata']]

        accuracy, _ = getCoverageStats(D_mov,analysisFieldList=columnsNeeded,capacityField=capacityField)

        D_time = D_mov.dropna(subset=columnsNeeded)

        plannedTime =  D_time[timeColumns['dischargingpta']] - D_time[timeColumns['loadingptd']]
        actualTime =   D_time[timeColumns['dischargingata']] -  D_time[timeColumns['loadingatd']]

        Los = actualTime<plannedTime
        D_res = Los.value_counts()

        fig1=plt.figure()
        plt.pie(D_res,autopct='%1.1f%%', shadow=True, startangle=90,labels=D_res.index)
        plt.title('Level of Service')

        output_figure['level_of_service']=fig1

        coverages=pd.DataFrame([accuracy])

    return output_figure, coverages
