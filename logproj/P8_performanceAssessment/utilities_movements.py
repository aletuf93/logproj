# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %%
def getCoverageStats(D_mov,analysisFieldList,capacityField='QUANTITY'):
    #ritorna le statistiche di copertura e il numero di elementi
    #le statistiche di copertura sono una tupla di due elementi,
    #indicanti la copertura percentuale sul numero di linee e le quantita' rispettivamente
    n_lines=len(D_mov)
    tot_qties = np.nansum(D_mov[capacityField])


    tot_lines=len(D_mov[analysisFieldList].dropna())

    #se vedo verificare una copertura su una lista di attributi
    if isinstance(analysisFieldList,list):
        listCol=analysisFieldList
        listCol.append(capacityField)
        D_filtered_qties=D_mov[listCol].dropna()

#se devo verificare la copertura su un singolo attributo
    else:
        D_filtered_qties=D_mov[[analysisFieldList,capacityField]].dropna()

    lineCoverage = tot_lines/n_lines

    #la copertura sulle quantita' funziona solo se analysisFieldList e capacityField
    # sono diversi, altrimenti somma due volte la stessa colonna
    if capacityField==analysisFieldList:
        qtyCoverage=1
    else:
        qtyCoverage =  np.nansum(D_filtered_qties[capacityField])/tot_qties

    return (lineCoverage,qtyCoverage), tot_lines




# %%
def movementStatistics(D_mov, capacityField='QUANTITY'):
    #this function performs global analysis on the D_mov dataframe
    #returning a dataframe with global statistics

    data={}
    coverage_stats={}


    for col in D_mov.columns:
        # per tutti calcolo le statistiche di conteggio
        coverage_stats[f"COUNT.{col}"], nrows = getCoverageStats(D_mov,col,capacityField)
        if any( [ isinstance(i ,dict) for i in D_mov[col]]) :
            data[f"COUNT.{col}"] = nrows
        else:
            data[f"COUNT.{col}"] = len(D_mov[col].unique())
        # se e' un numero calcolo le statistiche di somma
        if (D_mov[col].dtypes==np.float) |(D_mov[col].dtypes==np.int):
            data[f"SUM.{col}"] = np.nansum(D_mov[col])
            coverage_stats[f"SUM.{col}"] = coverage_stats[f"COUNT.{col}"]

        # se e' una data identifico il numero di giorni, il primo e l'ultimo giorno
        if (D_mov[col].dtypes==np.datetime64) |  (D_mov[col].dtypes=='<M8[ns]'):
            BookingDates = np.unique(D_mov[col].dt.date)
            beginningTimeHorizon=min(BookingDates)
            endTimeHorizon=max(BookingDates)
            NofBookingDays=len(BookingDates)

            data[f"N.OF.DAYS.{col}"]=NofBookingDays
            data[f"FIRST.DAY.{col}"]= beginningTimeHorizon
            data[f"LAST.DAY.{col}"]= endTimeHorizon

            coverage_stats[f"N.OF.DAYS.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"FIRST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"LAST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]


    D_global=pd.DataFrame([data,coverage_stats]).transpose()
    D_global.columns=['VALUE','ACCURACY']
    return  D_global
