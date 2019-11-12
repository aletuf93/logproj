import numpy as np
import pandas as pd

def create_MacroArea(df):

    df['Reparti']=''

    for i in np.arange(0, len(df['LocCodeX'].unique()) ,1):#ATTENZIONE A COME SI GESTISCE IL VALORE! SI RISCHIA DI PERDERE UN PEZZO

         list_ub = list(df['LocCodeX'].unique())
         list_ub.sort()
         [round(x) for x in list_ub]
         df.loc[(df['LocCodeX'] == list_ub[int(i)]) , 'Reparti'] = 'Reparto_'+(str(int(i)))

    df.dropna(subset = ['Reparti'])
    df_final = df.groupby('Reparti', as_index=False)['ACCESSI'].sum()
    df_final['Text'] = ''
    if (not df_final['Reparti'].empty) :
        df_final.loc[df_final['Reparti'].notnull(), 'Text'] = 'Area'
    return df, df_final
