import numpy as np
import pandas as pd
from scipy.stats import poisson
import logproj.stat_time_series as ts
from logproj.P1_familyProblem.part_classification import returnsparePartclassification


# %% SIMULATE INVENTORY FUNCTION
def returnProbabilisticInventory(I_t, iterations=30):
    '''


    Parameters
    ----------
    I_t : TYPE list
        DESCRIPTION. inventory function with an element for each day
    iterations : TYPE int
        DESCRIPTION. number of iterations to simulate the inventory function
    Returns
    -------
    min_real_I_t : TYPE int
        DESCRIPTION. min value from the real inventory function I_t
    max_real_I_t : TYPE int
        DESCRIPTION. max value from the real inventory function I_t
    avg_real_I_t : TYPE int
        DESCRIPTION. average value from the real inventory function I_t
    min_probabilistic_I_t : TYPE int
        DESCRIPTION. min value from the probabilistic inventory function
    max_probabilistic_I_t : TYPE int
        DESCRIPTION. max value from the probabilistic inventory function
    avg_probabilistic_I_t : TYPE int
        DESCRIPTION. average value from the probabilistic inventory function

    '''

    #service function
    def movementsGeneratorFromDemandPattern(demand_pattern, n_days, ADI, mean_interarrival_time, std_interarrival_time, mean_qty, sigma_qty):
    #observed demand pattern
    #n_days number of inventory days to simulate
    #ADI ADI value of the observed movements
    #mean_interarrival_time average number of days between two movements
    #std_interarrival_time std of the number of days between two movements
    #mean_qty average quantity of a single movement
    #sigma_qty std of the quantity of a single movement

    # if intermitten or lumpy, use a poisson to generate interarrival times
        if demand_pattern in(["",""]):

            #integer values are allowed due to Poisson distribution for out_days
            out_days = poisson.rvs(mu=ADI, size=n_days)

        #if stable or erratic, use a gaussian distribution to generate interarrival times
        elif demand_pattern in(["ERRATIC","STABLE","INTERMITTENT","LUMPY"]):

            #only 0/1 values are allowed for out_days
            out_days = np.zeros(n_days)
            dayPointer=0
            while dayPointer<len(out_days):
                waitingDays= int(np.round(np.random.normal(mean_interarrival_time, std_interarrival_time),0))
                dayPointer=dayPointer+waitingDays
                if dayPointer<len(out_days):
                    out_days[dayPointer]=1

        else:
            print("Error, demand pattern not found")

        #generates the demand values
        out_quantities = np.round(np.random.normal(mean_qty, sigma_qty,len(out_days)),0)

        #generates probabilistic outbound movements
        M_prob = out_quantities * out_days
        #plt.figure()
        #plt.plot(out_days)
        return M_prob

    #initialise output
    #compare simulated and real values
    min_real_I_t = max_real_I_t = avg_real_I_t =std_real_I_t=0
    min_probabilistic_I_t = max_probabilistic_I_t = avg_probabilistic_I_t=std_probabilistic_I_t=0


    I_t_cleaned = [x for x in I_t if str(x) != 'nan'] #remove nan inventories (e.g. at the beginning of the series if the part is not in the WH)


    #generate the movement function
    M_t = movementfunctionfromInventory(I_t_cleaned)

    # plot movement and inventory functions
    #plt.plot(M_t)
    #plt.plot(I_t_cleaned)

    #define the outbound quantity distribution
    M_t_out=M_t[M_t['QUANTITY']<0]
    if len(M_t_out)>0:
        #identify quantity parameters
        mean_qty_out = float(np.abs(np.mean(M_t_out)))
        sigma_qty_out = float(np.std(M_t_out))
        if sigma_qty_out==0: #having only an observation, use as sigma half of the mean
            sigma_qty_out = mean_qty_out/2

        #identify time parameters
        ADI_OUT =len(M_t_out)/len(M_t)
        interarrival_time = []
        for j in range(1,len(M_t_out)):
            interarrival_time.append(M_t_out.index[j] - M_t_out.index[j-1])
        mean_interarrival_time_out = np.mean(interarrival_time)
        std_interarrival_time_out = np.std(interarrival_time)

        if std_interarrival_time_out==0: #having only an observation, use as sigma half of the mean
            std_interavvial_time_out=mean_interarrival_time_out/2


        #depending on the demand pattern, simulates the inventory behaviour
        ADI = ADI_OUT
        CV2 = sigma_qty_out/mean_qty_out
        demand_pattern = returnsparePartclassification(ADI=ADI, CV2=CV2)

        #generates the days of the market demand
        min_inventory=[]
        max_inventory=[]
        avg_inventory=[]
        std_inventory=[]


        for iteration in range(0,iterations):
            #generate outbound movements
            M_prob_out = movementsGeneratorFromDemandPattern(demand_pattern = demand_pattern_out,
                  n_days = len(I_t),
                  ADI=ADI_OUT,
                  mean_interarrival_time = mean_interarrival_time_out,
                  std_interarrival_time = std_interarrival_time_out,
                  mean_qty=mean_qty_out,
                  sigma_qty=sigma_qty_out)

            #generate inbound movements
            M_prob_in = movementsGeneratorFromDemandPattern(demand_pattern = demand_pattern_in,
                          n_days = len(I_t),
                          ADI=ADI_IN,
                          mean_interarrival_time = mean_interarrival_time_in,
                          std_interarrival_time = std_interarrival_time_in,
                          mean_qty=mean_qty_in,
                          sigma_qty=sigma_qty_in)

            #generate inventory
            I_prob = [0]
            for i in range(0,len(M_prob_in)):
                I_prob.append(I_prob[i]+M_prob_in[i]-M_prob_out[i])
            

            I_prob = pd.DataFrame(I_prob,columns=['INVENTORY'])
            I_prob['INVENTORY'] = I_prob['INVENTORY']-min(I_prob['INVENTORY'])
            if len(I_prob.dropna())>0:
                I_prob['INVENTORY'] = I_prob['INVENTORY'].fillna(method='bfill') #defines the values of the inventory value
                I_prob['INVENTORY'] = I_prob['INVENTORY'].fillna(method='ffill') #fill nan values after the last outbound movement
                min_inventory.append(min(I_prob[I_prob['INVENTORY']>0]['INVENTORY'])) #ignore zero inventory values
                max_inventory.append(max(I_prob['INVENTORY']))
                avg_inventory.append(np.mean( I_prob['INVENTORY']))
                std_inventory.append(np.std( I_prob['INVENTORY']))
            #plt.plot(I_prob['INVENTORY'])

        #compare simulated and real values
        min_real_I_t = np.nanmin([j for j in I_t if j>0])
        max_real_I_t = np.nanmax(I_t)
        avg_real_I_t = np.nanmean(I_t)
        std_real_I_t = np.nanstd(I_t)

        min_probabilistic_I_t = np.nanmin(min_inventory)
        max_probabilistic_I_t = np.nanmax(max_inventory)
        avg_probabilistic_I_t = np.nanmean(avg_inventory)
        std_probabilistic_I_t = np.nanmean(std_inventory)



    #print(f"REAL: ({min_real_I_t},{avg_real_I_t},{max_real_I_t})")
    #print(f"OPTIMAL ({min_probabilistic_I_t},{avg_probabilistic_I_t},{max_probabilistic_I_t})")
    return min_real_I_t, max_real_I_t, avg_real_I_t, std_real_I_t, min_probabilistic_I_t, max_probabilistic_I_t, avg_probabilistic_I_t, std_probabilistic_I_t


# %% RETURN AND RESAMPLE MOVEMENTS

def returnResampleMovements(D_mov_loc):

    #Creo serie movimenti positivi
    D_mov_loc_positive = D_mov_loc[D_mov_loc['MOVEMENTS']>0]
    D_mov_loc_positive_series =  D_mov_loc_positive
    D_mov_loc_positive_series = D_mov_loc_positive_series.set_index('TIMESTAMP_IN',drop=True)
    D_mov_loc_positive_series = D_mov_loc_positive_series['MOVEMENTS']
    D_mov_loc_positive_series = D_mov_loc_positive_series.resample('1D').sum()
    D_mov_loc_positive_series = D_mov_loc_positive_series.to_frame()
    D_mov_loc_positive_series['PERIOD'] = D_mov_loc_positive_series.index
    D_mov_loc_positive_series['PERIOD'] = ts.sampleTimeSeries(D_mov_loc_positive_series['PERIOD'],'day')
    MOVEMENT_POSITIVE_DAYS = list(D_mov_loc_positive_series['PERIOD'])
    MOVEMENT_POSITIVE = list(D_mov_loc_positive_series['MOVEMENTS'])


    #Creo serie movimenti negativi
    D_mov_loc_negative = D_mov_loc[D_mov_loc['MOVEMENTS']<0]
    D_mov_loc_negative_series =  D_mov_loc_negative
    D_mov_loc_negative_series = D_mov_loc_negative_series.set_index('TIMESTAMP_IN',drop=True)
    D_mov_loc_negative_series = D_mov_loc_negative_series['MOVEMENTS']
    D_mov_loc_negative_series = D_mov_loc_negative_series.resample('1D').sum()
    D_mov_loc_negative_series = D_mov_loc_negative_series.to_frame()
    D_mov_loc_negative_series['PERIOD'] = D_mov_loc_negative_series.index
    D_mov_loc_negative_series['PERIOD'] = ts.sampleTimeSeries(D_mov_loc_negative_series['PERIOD'],'day')
    D_mov_loc_negative_series['MOVEMENTS'] = np.abs(D_mov_loc_negative_series['MOVEMENTS'])
    MOVEMENT_NEGATIVE_DAYS = list(D_mov_loc_negative_series['PERIOD'])
    MOVEMENT_NEGATIVE = list(D_mov_loc_negative_series['MOVEMENTS'])

    return MOVEMENT_POSITIVE_DAYS, MOVEMENT_POSITIVE, MOVEMENT_NEGATIVE_DAYS, MOVEMENT_NEGATIVE


# %%
def extractInventoryFromDataframe(D_loc, dateAttribute = 'INVENTORY_DAYS', listField = 'INVENTORY_QUANTITY'):
    D_inventory=pd.DataFrame([],columns=['GLOBAL_TREND_QUANTITY'])

    for i in range(0,len(D_loc)):
            #i=33159

            list_days = D_loc.iloc[i][dateAttribute]

            #go on only if an inventory has been saved
            if isinstance(list_days,list):

                list_inventory = np.array(D_loc.iloc[i][listField])
                list_inventory = np.nan_to_num(list_inventory) #convert nan to 0





                D_temp = pd.DataFrame(list_inventory,index=list_days,columns=['LOC_INVENTORY'] )
                D_inventory = pd.concat([D_temp, D_inventory], axis=1, sort=False)
                D_inventory=D_inventory.fillna(0)
                D_inventory['GLOBAL_TREND_QUANTITY'] = D_inventory['GLOBAL_TREND_QUANTITY'] + D_inventory['LOC_INVENTORY']
                D_inventory=D_inventory.drop(columns=['LOC_INVENTORY'])
    return D_inventory


# %%
def returnInventoryPart(D_movements, D_inventory, timeLineDays,quantityColums='QUANTITY'):
    '''
    Defines the inventory function (grouped by day) of a part

    Parameters
    ----------
    D_movements : TYPE: pandas dataframe
        DESCRIPTION. dataframe of movements of a single part (ITEMCODE)
    D_inventory : TYPE: pandas dataframe
        DESCRIPTION. dataframe of inventory of a single part (ITEMCODE) already grouped by TIMESTAMP and ITEMCODE
    timeLineDays : TYPE Pandas dataframe
        DESCRIPTION. dataframe wih a column TIMELINE having an aggregation of all the days to generate the inventory array
    quantityColums : TYPE string
        DESCRIPTION. indicates the column with the movement or inventory quantity
    Returns
    -------
    array_days : TYPE list
        DESCRIPTION. list of days where the inventory is reconstructed
    array_inventory : TYPE list
        DESCRIPTION. list of inventory values where the inventory is reconstructed

    '''

    #considero solo le righe con il segno
    D_movements=D_movements[D_movements['INOUT'].isin(['+','-'])]
    #se ho almeno un movimento
    if len(D_movements)>0:
        #identifico i segni dei movimenti
        D_movements['MOVEMENT'] = D_movements['INOUT'].astype(str) + D_movements[quantityColums].astype(str)
        D_movements['MOVEMENT']=D_movements['MOVEMENT'].astype(float)

        #raggruppo su base giornaliera
        D_movements['PERIOD'] = ts.sampleTimeSeries(D_movements['TIMESTAMP_IN'],'day')
        D_movements_grouped = D_movements.groupby(['PERIOD']).sum()['MOVEMENT'].reset_index()
        D_movements_grouped = D_movements_grouped.sort_values(by='PERIOD')

        #define the inventory, given the movements
        D_movements_grouped['INVENTORY']=np.nan
        D_movements_grouped.at[0,'INVENTORY'] = D_movements_grouped.iloc[0]['MOVEMENT']
        for i in range(1,len(D_movements_grouped)):
            D_movements_grouped.at[i,'INVENTORY']= D_movements_grouped.iloc[i-1]['INVENTORY'] + D_movements_grouped.iloc[i]['MOVEMENT']
        if min(D_movements_grouped['INVENTORY'])<0:
            D_movements_grouped['INVENTORY'] = D_movements_grouped['INVENTORY'] - min(D_movements_grouped['INVENTORY'])
        #se non ho movimenti setto a zero
    else:
        D_movements_grouped = pd.DataFrame([[timeLineDays['TIMELINE'].iloc[0],0]],columns=['PERIOD','INVENTORY'])

    #faccio join con la linea temporale
    D_inventory_part = timeLineDays.merge(D_movements_grouped, how='left', left_on='TIMELINE', right_on='PERIOD')

    #uso forward fill per riempire i nulli: se non ho inventari calcolati, fa fede l'ultimo inventario calcolato
    D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'].fillna(method='ffill')

    #riempio il resto con zeri
    D_inventory_part['INVENTORY'] =  D_inventory_part['INVENTORY'].fillna(0)


    #se ho almeno un punto di giacenza correggo la stima della curva di inventario
    if len(D_inventory)>0:
        D_inventory['PERIOD'] = ts.sampleTimeSeries(D_inventory['TIMESTAMP'],'day')
        D_inventory = D_inventory.merge(D_inventory_part, how='left', left_on='PERIOD', right_on='TIMELINE')

        #per ogni punto di giacenza noto tento di sistemare la curva di inventario
        for index, row in D_inventory.iterrows():

            #verifico se ho sottostimato il livello di inventario. Non posso aver
            # sorastimato perche' ho gia' portato tutto sopra lo zero
            if row[quantityColums] > row.INVENTORY:
                gap = row[quantityColums] - row.INVENTORY
                D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'] + gap


    array_days = list(D_inventory_part['TIMELINE'])
    array_inventory = list(D_inventory_part['INVENTORY'])

    return array_days, array_inventory




# %%
def movementfunctionfromInventory(I_t_cleaned):
    '''


    Parameters
    ----------
    I_t_cleaned : TYPE list
        DESCRIPTION. list of inventory values without nan values

    Returns
    -------
    M_t : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe of movements

    '''
    M_t=[]
    for j in range(1,len(I_t_cleaned)):
        M_t.append(I_t_cleaned[j]-I_t_cleaned[j-1])
    M_t=pd.DataFrame(M_t,columns=['QUANTITY'])
    return M_t

# %% measure the interarrival time between inbound activities

def assessInterarrivalTime(I_t):
    #remove nan values
    I_t_cleaned = [x for x in I_t if str(x) != 'nan'] #remove nan inventories (e.g. at the beginning of the series if the part is not in the WH)

    #generate the movement function
    M_t = movementfunctionfromInventory(I_t_cleaned)

    M_t_in=M_t[M_t['QUANTITY']>0]
    interarrival_time = []
    for j in range(1,len(M_t_in)):
        interarrival_time.append(M_t_in.index[j] - M_t_in.index[j-1])

    #if one or zero data point set the interarrival time equal to zero
    if len(M_t_in)<=1:
        interarrival_time.append(0)

    mean_interarrival_time_in = np.mean(interarrival_time)
    std_interarrival_time_in = np.std(interarrival_time)
    return mean_interarrival_time_in, std_interarrival_time_in, interarrival_time

# %%
def updatePartInventory(D_SKUs,D_movements,D_inventory,timecolumn_mov,itemcodeColumns_sku,itemcodeColumns_mov,itemcodeColumns_inv):

    D_SKUs['INVENTORY_QTY'] = [[] for i in range(0,len(D_SKUs))]
    D_SKUs['INVENTORY_DAYS'] = [[] for i in range(0,len(D_SKUs))]

    #  define the inventory quantity
    firstDay = min(D_movements[timecolumn_mov]).date()
    lastDay = max(D_movements[timecolumn_mov]).date()
    timeLine = pd.date_range(start=firstDay, end=lastDay).to_frame()
    timeLineDays= ts.sampleTimeSeries(timeLine[0],'day').to_frame()
    timeLineDays.columns=['TIMELINE']

    D_SKUs = D_SKUs.reset_index(drop=True)
    #  Build a daily inventory array for each part
    for index, row in D_SKUs.iterrows():
        #print(part)
        #part = list(set(D_mov['ITEMCODE']))[0]
        part = row[itemcodeColumns_sku]

        #filter movements by itemcode
        D_mov_part = D_movements[D_movements[itemcodeColumns_mov]==part]

        #filter inventory by itemcode
        D_inv_part = D_inventory[D_inventory[itemcodeColumns_inv]==part]

        array_days, array_inventory = returnInventoryPart(D_mov_part, D_inv_part, timeLineDays)
        #plt.plot(array_days,array_inventory)

        #update the dataframe
        D_SKUs.at[index,'INVENTORY_QTY'] = array_inventory
        D_SKUs.at[index,'INVENTORY_DAYS'] = array_days
    return D_SKUs

# %% UPDATE INVENTORY PARAMETERS

def updateInventoryParams(D_SKUs):

    #create result columns
    D_SKUs['INVENTORY_REAL_MIN']=np.nan
    D_SKUs['INVENTORY_REAL_MAX']=np.nan
    D_SKUs['INVENTORY_REAL_AVG']=np.nan
    D_SKUs['INVENTORY_REAL_STD']=np.nan
    D_SKUs['INVENTORY_PROB_MIN']=np.nan
    D_SKUs['INVENTORY_PROB_MAX']=np.nan
    D_SKUs['INVENTORY_PROB_AVG']=np.nan
    D_SKUs['INVENTORY_PROB_STD']=np.nan

    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            min_real_I_t, max_real_I_t, avg_real_I_t, std_real_I_t, min_probabilistic_I_t, max_probabilistic_I_t, avg_probabilistic_I_t, std_probabilistic_I_t = returnProbabilisticInventory(I_t)

            #update the dataframe

            D_SKUs.at[index,'INVENTORY_REAL_MIN']=min_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_MAX']=max_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_AVG']=avg_real_I_t
            D_SKUs.at[index,'INVENTORY_REAL_STD']=std_real_I_t
            D_SKUs.at[index,'INVENTORY_PROB_MIN']=min_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_MAX']=max_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_AVG']=avg_probabilistic_I_t
            D_SKUs.at[index,'INVENTORY_PROB_STD']=std_probabilistic_I_t


    return D_SKUs


# %% UPDATE INTERARRIVAL TIME

def updateInterarrivalTime(D_SKUs):

    #create result columns
    D_SKUs['INTERARRIVAL_MEAN_IN']=np.nan
    D_SKUs['INTERARRIVAL_STD_IN']=np.nan


    for index, row in D_SKUs.iterrows():
        #select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        #calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements=movements.dropna()
        if len(movements)>0:
            mean_interarrival_time_in, std_interarrival_time_in, _ = assessInterarrivalTime(I_t)

            #update the dataframe

            D_SKUs.at[index,'INTERARRIVAL_MEAN_IN']=mean_interarrival_time_in
            D_SKUs.at[index,'INTERARRIVAL_STD_IN']=std_interarrival_time_in

    return D_SKUs
