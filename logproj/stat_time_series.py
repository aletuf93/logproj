# %% import packages

import numpy as np
import pandas as pd
import itertools
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from pandas.api.types import CategoricalDtype

from scipy.stats import boxcox


# %% MANIPULATE DATETIME

def timeStampToDays(series):
    # converto da una pandas timestamp a un numero in giorni
    D = series.dt.components['days']
    H = series.dt.components['hours']
    M = series.dt.components['minutes']
    result = D + (H / 24) + (M / (60 * 24))
    return result


def sampleTimeSeries(series, sampleInterval):
    '''
    sample a pandas series using a sampling interval
    sampleInterval is the sampling interval od D_mov: day, week, month, year
    '''
    if sampleInterval == 'day':
        series = series.dt.strftime('%Y-%j')
    elif sampleInterval == 'week':
        series = series.dt.strftime('%Y-%U')
    elif sampleInterval == 'month':
        series = series.dt.strftime('%Y-%m')
    elif sampleInterval == 'year':
        series = series.dt.strftime('%Y')
    return series


def raggruppaPerSettimana(df, timeVariable, groupVariable, tipo):

    if isinstance(df, pd.Series):  # convert to dataframe if a series
        df = pd.DataFrame([[df.index.values.T, df.values]],
                          columns=[timeVariable, groupVariable])

    df['DatePeriod'] = pd.to_datetime(df[timeVariable]) - pd.to_timedelta(7, unit='d')

    if tipo == 'count':
        df = df.groupby([pd.Grouper(key=timeVariable,
                                    freq='W-MON')])[groupVariable].size()
    elif tipo == 'sum':
        df = df.groupby([pd.Grouper(key=timeVariable,
                                    freq='W-MON')])[groupVariable].sum()
    df = df.sort_index()
    return df


def raggruppaPerMese(df, timeVariable, groupVariable, tipo):

    if isinstance(df, pd.Series):  # convert to dataframe if a series
        df = pd.DataFrame([[df.index.values.T, df.values]],
                          columns=[timeVariable, groupVariable])

    # df['DatePeriod'] = pd.to_datetime(df[timeVariable]) - pd.to_timedelta(7, unit='d')

    if tipo == 'count':
        df = df.groupby([pd.Grouper(key=timeVariable, freq='M')])[groupVariable].size()
    elif tipo == 'sum':
        df = df.groupby([pd.Grouper(key=timeVariable, freq='M')])[groupVariable].sum()
    df = df.sort_index()
    return df


def raggruppaPerGiornoDellaSettimana(df, timeVariable, seriesVariable):
    cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cat_type = CategoricalDtype(categories=cats, ordered=True)

    df['Weekday'] = df[timeVariable].dt.day_name()
    df['Weekday'] = df['Weekday'].astype(cat_type)
    D_grouped = df.groupby(['Weekday']).agg({seriesVariable: ['size', 'mean', 'std']})
    D_grouped.columns = D_grouped.columns.droplevel(0)
    D_grouped['mean'] = np.round(D_grouped['mean'], 2)
    D_grouped['std'] = np.round(D_grouped['std'], 2)
    return D_grouped


def assegnaGiornoSettimana(df, dateperiodColumn):
    dayOfTheWeek = df[dateperiodColumn].dt.weekday_name
    weekend = (dayOfTheWeek == 'Sunday') | (dayOfTheWeek == 'Saturday')
    weekEnd = weekend.copy()
    weekEnd[weekend] = 'Weekend'
    weekEnd[~weekend] = 'Weekday'
    return dayOfTheWeek, weekEnd


# In[1]: ACF and PACF
def ACF_PACF_plot(series):
    # the function return the graph wit h a time series, the ACF and the PACF
    # it rreturns two arrays with the significant lags in the ACF and PACF

    fig = plt.subplot(131)

    plt.plot(series, 'skyblue')
    plt.xticks(rotation=30)
    plt.title('Time Series')

    lag_acf = acf(series, nlags=20)
    lag_pacf = pacf(series, nlags=20)

    plt.subplot(132)
    plt.stem(lag_acf, linefmt='skyblue', markerfmt='d')
    plt.axhline(y=0, linestyle='--')
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.title('ACF')
    plt.xlabel('time lag')
    plt.ylabel('ACF value')

    plt.subplot(133)
    plt.stem(lag_pacf, linefmt='skyblue', markerfmt='d')
    plt.axhline(y=0, linestyle='--')
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.title('PACF')
    plt.xlabel('time lag')
    plt.ylabel('PACF value')

    # identify significant values for ACF
    D_acf = pd.DataFrame(lag_acf, columns=['ACF'])
    D_acf['ORDER'] = D_acf.index.values + 1

    min_sign = -1.96 / np.sqrt(len(series))
    max_sign = 1.96 / np.sqrt(len(series))

    D_acf['SIGNIFICANT'] = (D_acf['ACF'] > max_sign) | (D_acf['ACF'] < min_sign)
    D_acf_significant = D_acf['ORDER'][D_acf['SIGNIFICANT']].values

    # identify significant values for PACF
    D_pacf = pd.DataFrame(lag_pacf, columns=['PACF'])
    D_pacf['ORDER'] = D_pacf.index.values + 1

    D_pacf['SIGNIFICANT'] = (D_pacf['PACF'] > max_sign) | (D_pacf['PACF'] < min_sign)
    D_pacf_significant = D_pacf['ORDER'][D_pacf['SIGNIFICANT']].values

    return fig, D_acf_significant, D_pacf_significant


def returnsignificantLags(D_pacf_significant, D_acf_significant, maxValuesSelected=2):
    # this function returns tuples of significant order (p, d, q) based on the lags of the function ACF_PACF_plot
    # select values for parameter p
    if len(D_pacf_significant) > 1:
        numSelected = min(maxValuesSelected, len(D_pacf_significant))
        p = D_pacf_significant[0: numSelected]

    else:
        p = [0, 1]

    # select values for parameter q
    if len(D_acf_significant) > 1:
        numSelected = min(maxValuesSelected, len(D_acf_significant))
        q = D_acf_significant[0: numSelected]
    else:
        q = [0, 1]

    d = [0, 1]
    a = [p, d, q]
    params = list(itertools.product(*a))
    return params

# %% ROLLING AVERAGE


def detrendByRollingMean(series, seasonalityPeriod):
    rolling_mean = series.rolling(window=seasonalityPeriod).mean()
    detrended = series.Series - rolling_mean
    return detrended


# %% ARIMA MODELS

# use this!
def SARIMAXfit(stationary_series, params):
    # this function tries different SARIMAX fits using tuples of orders specified in the list of tuples (p,d,q) param
    # on the time series stationary_series
    # the function return a figure_forecast with the plot of the forecast
    # a figure_residuals with the plot of the residuals
    # a dict resultModel with the model, the error (AIC), the order p,d,q
    '''
    #PACF=>AR
    #ACF=>MA
    #ARIMA(P,D,Q) = ARIMA(AR, I, MA)
    
    '''
    
    incumbentError = 999999999999999999999
    bestModel = []
    
    for param in params:
        mod = sm.tsa.statespace.SARIMAX(stationary_series,
                                        order=param,
                                        enforce_stationarity=True,
                                        enforce_invertibility=True,
                                        initialization='approximate_diffuse')
      
        results = mod.fit()
        if(results.aic < incumbentError):
            bestModel = mod
            incumbentError = results.aic
    
    results = bestModel.fit()
    figure_residuals = results.plot_diagnostics(figsize=(15, 12))
    
    figure_forecast = plt.figure()
    plt.plot(stationary_series)
    plt.plot(results.fittedvalues, color='red')
    plt.title('ARIMA fit p=' + str(bestModel.k_ar) + ' d=' + str(bestModel.k_diff) + ' q=' + str(bestModel.k_ma))
    
    resultModel = {'model': bestModel,
                   'aic': incumbentError,
                   'p': bestModel.k_ar,
                   'd': bestModel.k_diff,
                   'q': bestModel.k_ma}
    
    return figure_forecast, figure_residuals, resultModel


def ARIMAfit(series, p, d, q):
    # series=series[~np.isnan(series)]
    model = ARIMA(series, order=(p, d, q))
    results_AR = model.fit(disp=-1)
    plt.plot(series)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('ARIMA fit p=' + str(p) + ' q=' + str(q) + ' d=' + str(d))

    plt.figure()
    results_AR.plot_diagnostics(figsize=(15, 12))
    return 1


def forecastSARIMAX(addSeries, minRangepdq, maxRangepdqy, seasonality, NofSteps, titolo):
    NofSteps = np.int(NofSteps)
    # residui=plt.figure()
    result = autoSARIMAXfit(addSeries, minRangepdq, maxRangepdqy, seasonality)
    results = result.fit()
    residui = results.plot_diagnostics(figsize=(15, 12))

    forecast = plt.figure()
    pred = results.get_prediction(start=len(addSeries) - 1,
                                  end=len(addSeries) + NofSteps,
                                  dynamic=True)
    pred_ci = pred.conf_int()

    ax = addSeries.plot(label='observed', color='orange')
    pred.predicted_mean.plot(ax=ax, label='Dynamic forecast', color='r', style='--', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='y', alpha=.2)

    ax.set_xlabel('Timeline')
    ax.set_ylabel('Series value')
    plt.title('Forecast: ' + titolo)
    plt.legend()
    return residui, forecast

# In[1]: autofit


def autoSARIMAXfit(y, minRangepdq, maxRangepdqy, seasonality):
    minRangepdq = np.int(minRangepdq)
    maxRangepdqy = np.int(maxRangepdqy)
    seasonality = np.int(seasonality)

    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(minRangepdq, maxRangepdqy)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    incumbentError = 9999999999
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                if(results.aic < incumbentError):
                    bestModel = mod
                    incumbentError = results.aic

                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except Exception:
                continue
    return bestModel

# %% FOURIER ANALYSIS


def fourierAnalysis(y):

    y = y.reshape(len(y),)
    N = len(y)  # 600 campioni
    T = 1  # un campione alla settimana

    # plt.figure()
    t = np.arange(0, len(y)).reshape(len(y),)
    p = np.polyfit(t, y, 1)         # find linear trend in x
    y_notrend = y - p[0] * t
    # plt.plot(x,y_notrend)
    # plt.title('detrended signal')
    # plt.xlabel('settimane')

    # calcolo fourier transform
    yf = np.fft.fft(y_notrend)

    # filtro i valori più significativi (solo le frequenze il cui picco spiega almeo il 10% della stagionalità)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])
    weeks = 1 / xf

    # plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.grid()
    # plt.show()

    data = {'Frequency_domain_value': xf,
            'Time_domain_value': weeks,
            'Amplitude': amplitude}
    D = pd.DataFrame(data)
    D = D.replace([np.inf, -np.inf], np.nan)
    D = D.dropna()
    D = D.sort_values(['Amplitude'], ascending=False)
    D['perc'] = D['Amplitude'] / np.sum(D['Amplitude'])
    D['cumsum'] = D['perc'].cumsum()
    # D['Settimana']=np.round(D.Settimana,0)
    return D


# %% STATIONARITY TRANSFORM


def transformSeriesToStationary(timeSeries_analysis, signifAlpha=0.05):
    # this function tries log, power and square root transformation to stationary series
    # it returns the series and a string with the model used to transform the series

    # timeSeries_analysis is a pandas series to make stationary
    # signifAlpha is the significance level (0.1 , 0.05, 0.01) to accept or reject the null hypothesis of Dickey fuller

    # reference: http://www.insightsbot.com/blog/1MH61d/augmented-dickey-fuller-test-in-python
    def returnPandPstar(result):
        p_value = result[1]

        p_star = signifAlpha

        # in alternativa si puo' usare il valore della statistica del test e i valori critici
        '''
        if signifAlpha==0.01:
            p_star=result[4]['1%']
        elif signifAlpha==0.05:
            p_star=result[4]['5%']
        if signifAlpha==0.1:
            p_star=result[4]['10%']
        '''

        return p_value, p_star

    ###########################################################################
    # test the original series
    result = adfuller(timeSeries_analysis, autolag='AIC')
    p_value, p_star = returnPandPstar(result)

    '''
    If the P-Value is less than the Significance Level defined,
    we reject the Null Hypothesis that the time series contains a unit root.
    In other words, by rejecting the Null hypothesis,
    we can conclude that the time series is stationary.
    '''

    if (p_value < p_star):
        print("The initial series is stationary")
        model = 'initial'
        return timeSeries_analysis, model

    ###########################################################################
    # trying with power transformation
    timeSeries_analysis_transformed = timeSeries_analysis**2

    result = adfuller(timeSeries_analysis_transformed, autolag='AIC')
    p_value, p_star = returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using POWER transformation is stationary")
        model = 'POWER:2'
        return timeSeries_analysis_transformed, model

    ###########################################################################
    # trying with square root transformation
    timeSeries_analysis_transformed = np.sqrt(timeSeries_analysis)

    result = adfuller(timeSeries_analysis_transformed, autolag='AIC')
    p_value, p_star = returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using SQUARE ROOT transformation is stationary")
        model = 'SQRT'
        return timeSeries_analysis_transformed, model

    ###########################################################################
    # trying with logarithm transformation
    timeSeries_analysis_temp = timeSeries_analysis + 0.001
    timeSeries_analysis_transformed = np.log(timeSeries_analysis_temp)

    result = adfuller(timeSeries_analysis_transformed, autolag='AIC')
    p_value, p_star = returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using LOG transformation is stationary")
        model = 'LOG'
        return timeSeries_analysis_transformed, model

    ###########################################################################
    # trying with boxcox transformation
    timeSeries_analysis_transformed, lam = boxcox(timeSeries_analysis_temp)

    result = adfuller(timeSeries_analysis_transformed, autolag='AIC')
    p_value, p_star = returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using BOXCOX, lambda:{lam} transformation is stationary")
        model = f"BOXCOX, lambda:{lam}"
        return timeSeries_analysis_transformed, model

    print("No valid transformation found")
    return [], []
# %%


def attractor_estimate(y, dim='3d'):
    """
    Uses the Ruelle & Packard method to estimate an attractor

    Parameters
    ----------
    y : TYPE array
        DESCRIPTION. time series to evaluate
    dim : TYPE, optional string
        DESCRIPTION. The default is '3d'. '3d' or '2d' projection

    Returns
    -------
    None.

    """
    # TODO: add the time lag choice
    output_fig = {}
    #  Ruelle & Packard reconstruction
    y_2 = y[1:]
    y_3 = y[2:]
    
    # fix array length
    y = y[:len(y_3)]
    y_2 = y_2[:len(y_3)]
    
    if dim == '3d':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(y, y_2, y_3, lw=0.5)
        plt.title(f" {dim} attractor estimate")
        output_fig['attractor_fig'] = fig
    elif dim == '2d':
        fig = plt.figure()
        plt.plot(y, y_2, lw=0.5)
        plt.title(f" {dim} attractor estimate")
        output_fig['attractor_fig'] = fig
    else:
        print("Choose 3d or 2d dimension")


def poincare_section(series, T=2, num_of_dots_on_picture=10):
    """
    Define the poincare section of a time series at time lags T and output
    a figure for each time lag containing a given number of dots

    Parameters
    ----------
    series : TYPE array
        DESCRIPTION. time series to analyse
    T : TYPE, optional int
        DESCRIPTION. The default is 2. time lag at which evaluate the time series
    num_of_dots_on_picture : TYPE, optional int
        DESCRIPTION. The default is 10. number of dots for each image of the poincare section


    Returns
    -------
    D_all_coords : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe with poincare section coordinates for each time lag evaluated,
                     corresponding predicted value (next time lag()) and an image (rgb array) with the
                     num_of_dots_on_picture poincare section evaluated at that step
    out_fig : TYPE dict
        DESCRIPTION. cistionary containing the poincare section at the last time lag

    """
    # create an output dictionary for figures
    out_fig = {}
    
    # create a dataframe with coordinates of the poincare section
    # the corrensponding predicting value
    D_all_coords = pd.DataFrame(columns=['x_coord', 'y_coord', 'value_to_predict'])
    
    # define the poincare section at each time lag
    for i in range(T, len(series) - 1):
        poincare_new_coord = (series[i], series[i - T], series[i + 1])
        D_all_coords = D_all_coords.append(pd.DataFrame([poincare_new_coord],
                                                        columns=['x_coord', 'y_coord', 'value_to_predict']))
    
    # set progressive index
    D_all_coords.index = list(range(0, len(D_all_coords)))
    
    # plot Poincare Section of the Time series with the given Time Lag
    
    # set colors
    c_list = list(range(len(D_all_coords)))
    cmap = cm.autumn
    norm = Normalize(vmin=min(c_list), vmax=max(c_list))
    
    # define the figure
    fig = plt.figure()
    plt.scatter(D_all_coords['x_coord'], D_all_coords['y_coord'], s=0.5, c=cmap(norm(c_list)))
    plt.title(f"Poincare section with k={T}")
    out_fig['PoincareSection'] = fig
    
    # output the image arrays for predictions
    
    # add a column for the images with the poincare sections
    D_all_coords['PoincareMaps'] = ''
    for position in range(0, len(D_all_coords)):
        
        beginning = max(0, position - num_of_dots_on_picture)
        end = position + 1
        plt.scatter(D_all_coords['x_coord'].iloc[beginning:end], D_all_coords['y_coord'].iloc[beginning:end], s=0.5, c='black')
        plt.xlim((min(D_all_coords['x_coord']), max(D_all_coords['x_coord'])))
        plt.ylim((min(D_all_coords['y_coord']), max(D_all_coords['y_coord'])))
        plt.axis('off')
        out_fig['PoincareSection'] = fig
        fig.canvas.draw()
        
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        D_all_coords.at[position, 'PoincareMaps'] = data
    
    return D_all_coords, out_fig
# %%
# TODO: mov examples into ipynb notebook


'''
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=2.667):
    ''' '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    ''' '''
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

# %%

x = list(np.arange(-10, 10, 0.11))
y = np.sin(x)
plt.figure()
plt.plot(y)
plt.title('series')

attractor_estimate(xs, dim='2d')
attractor_estimate(xs, dim='3d')

poincare_section(xs[0:100], T=10, num_of_dots_on_picture=100)
'''
