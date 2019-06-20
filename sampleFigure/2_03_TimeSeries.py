# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:11:25 2019

@author: Alessandro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import STAT_timeSeries as ts
import sklearn.linear_model as lm
from statsmodels.tsa.stattools import adfuller



# In[1]: Additive and multiplicative time series (Figure 10)
numeroRecord=50
x=np.arange(0,numeroRecord)
y1=x+3
y2=np.cos(x)*20
y3=np.random.randn(1, numeroRecord)



#  additive Time series
y=y1+y2+y3
y=np.transpose(y)
addSeries = pd.DataFrame(y, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.subplot(121) 
plt.plot(addSeries.Series,'--',color='skyblue')
plt.xticks(rotation=30)
plt.title('Additive Time Series')

# multiplicative Time series
numeroRecord=50
x=np.arange(0,numeroRecord)
y1=x*2
y2=abs(np.cos(x)*20 + 1)
y3=1
y=y1*y2*y3
y=np.transpose(y)
mulSeries = pd.DataFrame(y, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.subplot(122) 
plt.plot(mulSeries.Series,'k--',color='skyblue')
plt.xticks(rotation=30)
plt.title('Multiplicative Time Series')


# In[1]: additive Time series transposed with ACF and PACF (figure 4)
numeroRecord=40
x=np.arange(0,numeroRecord)
y1=x*2.8 + 4.3
y2=np.cos(20*x)*39
y3=np.random.randn(1, numeroRecord)


y=y1+y2+y3
y=np.transpose(y)
y=y+6000

addSeries = pd.DataFrame(y, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
ts.ACF_PACF_plot(addSeries)




# In[1]: time series decomposition trend (Figure 11)
numeroRecord=50
x=np.arange(0,numeroRecord)
y1=x+3
y2=np.cos(x)*20
y3=np.random.randn(1, numeroRecord)



# define the additive Time series
y=y1+y2+y3
y=np.transpose(y)
addSeries = pd.DataFrame(y, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])

regr = lm.LinearRegression()
x=x.reshape(-1,1)
y=y.reshape(-1,1)
lr=regr.fit(x,y)
y_pred=lr.predict(x)

plt.subplot(121) 
trend = pd.DataFrame(y_pred, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.plot(trend.Series,'k--',color='skyblue')
plt.xticks(rotation=30)
plt.title('Trend component extracted by OLS')

#traccio i residui
plt.subplot(122) 
residuals=y-y_pred
trend = pd.DataFrame(residuals, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.plot(trend.Series,'k--',color='skyblue')
plt.xticks(rotation=30)
plt.title('Residuals: seasonality + random')

# In[1]: seasonal component averaging (Figure 13)

season_frequency=7
residuals_S=residuals

#tappo eventuali buchi con NaN
numeroRecord=len(y)
divisibile=season_frequency-numeroRecord%season_frequency
if divisibile!=0:
    for i in range(0,divisibile):
        residuals_S=np.append(residuals_S,np.nan)
        
numeroColonne= int(len(residuals_S)/season_frequency)  
residuals_S=residuals_S.reshape(season_frequency,numeroColonne)

season=np.nanmean(residuals_S, axis=1)
season=np.tile(season,numeroColonne)
for i in range(0,divisibile):
        season=season[0:-1]
        
plt.subplot(121) 
season = pd.DataFrame(season, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.plot(season.Series,'k--',color='skyblue')
plt.xticks(rotation=30)
plt.title('Seasonal component')  

#traccio i residui
plt.subplot(122) 
residuals=residuals-season
residuals = pd.DataFrame(residuals, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
plt.scatter(residuals.Series.index,residuals.Series.values, marker='o',color='orange')
plt.xticks(rotation=30)
plt.title('Seasonal component - residuals')  


# In[1]: seasonal component averaging (Figure 14 & 15)

numeroRecord=50
x=np.arange(0,numeroRecord)
y1=0.2*x+3
y2=np.cos(0.005*x)*20
y3=np.random.randn(1, numeroRecord)

y=y1+y2+y3
y=np.transpose(y)
#y=y+6000
addSeries = pd.DataFrame(y, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])
ts.ACF_PACF_plot(addSeries)


'''
# In[1]: detrend by moving average
rolling_mean = addSeries.Series.rolling(window = 6).mean()
detrended = addSeries.Series - rolling_mean
detrended.dropna()
plt.figure()
ts.ACF_PACF_plot(detrended)
'''

# In[1]: detrend and transform for stationarity

regr = lm.LinearRegression()
x=x.reshape(-1,1)
y=y.reshape(-1,1)
lr=regr.fit(x,y)
y_pred=lr.predict(x)
residuals=y-y_pred
residuals=residuals-min(residuals)
residuals=pd.DataFrame(residuals, index = pd.date_range('7/1/2016', freq = 'w', periods = numeroRecord), columns = ['Series'])


result=adfuller(residuals.iloc[:,0].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1]<0.05:
    print('The series is stationary')
else:
    print('The series is not stationary')
    
ts.ACF_PACF_plot(residuals)


#trasformazione potenza per rendere stazionaria
log_series = residuals.apply(lambda x: x**.5)
plt.figure()
log_series.plot()


ts.ACF_PACF_plot(log_series)

#test for stationarity
result=adfuller(log_series.iloc[:,0].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1]<0.05:
    print('The series is stationary')
else:
    print('The series is not stationary')



# In[1]: apply ARIMA
addSeries=log_series
result=ts.autoSARIMAXfit(addSeries,0,2,6)
results=result.fit()
results.plot_diagnostics(figsize=(15, 12))
plt.show()


pred = results.get_prediction(start=len(addSeries)-1,end=len(addSeries)+5, dynamic=True)
pred_ci = pred.conf_int()

ax = addSeries.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Dynamic forecast', color='r', style='--', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='y', alpha=.2)

ax.set_xlabel('Timeline')
ax.set_ylabel('Series value')
plt.legend()

plt.show()


# In[1]: trasformazione potenza
ts.fourierAnalysis(addSeries.Series.values)
