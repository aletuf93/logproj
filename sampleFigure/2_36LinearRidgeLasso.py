#!/usr/bin/env python
# coding: utf-8


# In[21]:
#genero campione statistico da analizzare
def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
sampledim=10
testsize=0.1
xs = randrange(sampledim, 50, 100)
ys = randrange(sampledim, 0, 100)
zs = 2*xs*ys*ys +xs

X=[xs, ys]
y= zs



fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[0],X[1], y,  c='orange',marker='o')
plt.title('Dataset scatterplot')



print('MEDIA X: '+str(np.round(np.mean(X),2)))
print('STD X: '+str(np.round(np.std(y),2)))



# In[19]:
#Preprocessing dei dati (centro e normalizzo i PREDITTORI e Y)

from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_scaled=np.vstack((preprocessing.scale(xs),preprocessing.scale(ys)))
X_scaled=np.transpose(X_scaled)

y_scaled = preprocessing.scale(y)
#X_scaled=np.transpose(X)
#y_scaled=y

fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X_scaled[:,0], X_scaled[:,1], y_scaled,  c='orange',marker='o')
plt.title('Dataset centered scatterplot')

# In[19]:
#Seleziono a caso il training e il testing set
#test_size indica la dimensione in percentuale del test set rispetto al trainSet
#random state is the random number generator used to randomly select the train/test set
from sklearn.model_selection import train_test_split
testsize=0.30;

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=testsize, random_state=42)


fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')
plt.title('Dataset training and testing set')



# In[20]:


# Fit Ordinary Least Squares: OLS
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import pandas as pd

#Costruisco regressione lineare sul train-set
regr = lm.LinearRegression()
lr=regr.fit(X_train, y_train)

#Testo la regressione lineare sul test-set
y_pred = lr.predict(X_test)
mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
beta=np.round(regr.coef_,2)

#Traccio il grafico
fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')

from matplotlib import cm

xx, yy = np.meshgrid(X_scaled[:,0], X_scaled[:,1])
exog = pd.core.frame.DataFrame({'TV':xx.ravel(),'Radio':yy.ravel()})
out = lr.predict(exog)
ax.plot_surface(xx, yy, out.reshape(xx.shape),cmap='binary', alpha=0.5)
plt.title('Ordinary least square')
ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')






# In[26]:


# Fit Ridge Regression (L2)
alpha_ridge = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_ridge)):
    #Costruisco ridge regression sul train-set
    ridge = lm.Ridge(alpha=alpha_ridge[i])
    rr=ridge.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = rr.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
    #print("lambda =", alpha_ridge[i])
    #print("R-squared =", r2)
    #print("Coefficients =", beta)
    

    #Traccio il grafico
    fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
    ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')


    xx, yy = np.meshgrid(X_scaled[:,0], X_scaled[:,1])
    exog = pd.core.frame.DataFrame({'TV':xx.ravel(),'Radio':yy.ravel()})
    out = rr.predict(exog)
    ax.plot_surface(xx, yy, out.reshape(xx.shape), color='blue', rstride=1,  cstride=1,alpha = 0.1)
    plt.title('Ridge Regression model lambda='+str(alpha_ridge[i]))
    ax.set_zlim3d(-3,3)
  


# In[36]:


# Fit Lasso Regression (L1)
alpha_lasso = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_lasso)):
    #Costruisco ridge regression sul train-set
    lasso = lm.Lasso(alpha=alpha_lasso[i])
    lar=lasso.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = lar.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
    #print("lambda =", alpha_lasso[i])
    #print("R-squared =", metrics.r2_score(y_test, y_pred))
    #print("Coefficients =", lar.coef_)
    #print("Intercept =", lar.intercept_)

    #Traccio il grafico del modello
    fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
    ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')


    xx, yy = np.meshgrid(X_scaled[:,0], X_scaled[:,1])
    exog = pd.core.frame.DataFrame({'TV':xx.ravel(),'Radio':yy.ravel()})
    out = lar.predict(exog)
    ax.plot_surface(xx, yy, out.reshape(xx.shape), color='yellow', rstride=1,  cstride=1,alpha = 0.1)
    plt.title('Lasso Regression model lambda='+str(alpha_lasso[i]))
    ax.set_zlim3d(-3,3)
    


# In[ ]:


# Fit Elastic-Net Regression (L1)
alpha_en = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_lasso)):
    #Costruisco ridge regression sul train-set
    en = lm.ElasticNet(alpha=alpha_en[i], l1_ratio=0.5)
    lar=en.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = lar.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
   

    #Traccio il grafico del modello
    fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
    ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')


    xx, yy = np.meshgrid(X_scaled[:,0], X_scaled[:,1])
    exog = pd.core.frame.DataFrame({'TV':xx.ravel(),'Radio':yy.ravel()})
    out = lar.predict(exog)
    ax.plot_surface(xx, yy, out.reshape(xx.shape), color='green', rstride=1,  cstride=1,alpha = 0.1)
    plt.title('Elastic Net Regression model lambda='+str(alpha_lasso[i]))
    ax.set_zlim3d(-3,3)


# In[ ]:

#Costruisco regressione lineare LARS sul train-set
regr = lm.Lars()
lr=regr.fit(X_train, y_train)

#Testo la regressione lineare sul test-set
y_pred = lr.predict(X_test)
mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
beta=np.round(regr.coef_,2)



#Traccio il grafico del modello
fig=figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], y_test,  c='g',marker='o')
ax.scatter3D(X_train[:,0], X_train[:,1], y_train,  c='r',marker='o')


xx, yy = np.meshgrid(X_scaled[:,0], X_scaled[:,1])
exog = pd.core.frame.DataFrame({'TV':xx.ravel(),'Radio':yy.ravel()})
out = lr.predict(exog)
ax.plot_surface(xx, yy, out.reshape(xx.shape), color='red', rstride=1,  cstride=1,alpha = 0.1)
plt.title('Least Angle Regression='+str(alpha_lasso[i]))
ax.set_zlim3d(-3,3)








#####################ESEMPIO IN R2#########################################
# In[21]:
#genero campione statistico da analizzare

import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
sampledim=14
testsize=0.1
X=np.arange(sampledim).reshape(sampledim,1)
rnd=np.random.random_sample(sampledim).reshape(sampledim,1)
y=50 + 2*X+rnd*30

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X, y,  color='blue')
plt.title('Dataset scatterplot')

plt.grid()
plt.axhline(linewidth=2, color='r') #evidenzio asse Ox
plt.axvline(linewidth=2, color='r') #evidenzio asse Oy

X_med=np.round(np.mean(X),2)
X_std=np.round(np.std(y),2)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
textstr = 'Mean(X)='+str(X_med)+' Std(X)='+str(X_std)   
plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)

# In[19]:
#Preprocessing dei dati (centro e normalizzo i PREDITTORI e Y)

from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
y_scaled = preprocessing.scale(y)


figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_scaled, y_scaled,  color='blue')
plt.title('Standardized dataset scatterplot')

plt.grid()
plt.axhline(linewidth=2, color='r') #evidenzio asse Ox
plt.axvline(linewidth=2, color='r') #evidenzio asse Oy

X_med=np.round(np.mean(X_scaled),2)
X_std=np.round(np.std(X_scaled),2)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
textstr = 'Mean(X)='+str(X_med)+' Std(X)='+str(X_std)   
plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)

# In[19]:
#Seleziono a caso il training e il testing set
#test_size indica la dimensione in percentuale del test set rispetto al trainSet
#random state is the random number generator used to randomly select the train/test set
from sklearn.model_selection import train_test_split
testsize=0.30;
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=testsize, random_state=42)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_test, y_test,  color='green')
plt.scatter(X_train, y_train,  color='blue')

#evidenzio gli assi
plt.grid()
plt.axhline(linewidth=2, color='r') #evidenzio asse Ox
plt.axvline(linewidth=2, color='r') #evidenzio asse Oy

#aggiungo didascalia su percentuale train-test
testDim=np.round(testsize*100,2)
trainDim=np.round((1-testsize)*100,2)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
textstr = 'Training set='+str(trainDim)+'% Testing set='+str(testDim) +'%'  
plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)

# In[20]:


# Fit Ordinary Least Squares: OLS
import sklearn.linear_model as lm
import sklearn.metrics as metrics

#Costruisco regressione lineare sul train-set
regr = lm.LinearRegression()
lr=regr.fit(X_train, y_train)

#Testo la regressione lineare sul test-set
y_pred = lr.predict(X_test)
mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
beta=np.round(regr.coef_,2)



#Traccio il grafico del modello
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_test, y_test,  color='black')
plt.scatter(X_train, y_train,  color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression model (OLS)')
plt.legend(['linear model','testing set','training set'])

#evidenzio gli assi
plt.grid()
plt.axhline(linewidth=2, color='r') #evidenzio asse Ox
plt.axvline(linewidth=2, color='r') #evidenzio asse Oy

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
textstr = 'MSE='+str(mse)+' \n Beta='+str(beta)   
plt.text(0.1, -1, textstr, fontsize=14, verticalalignment='top', bbox=props)



# In[26]:


# Fit Ridge Regression (L2)
alpha_ridge = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_ridge)):
    #Costruisco ridge regression sul train-set
    ridge = lm.Ridge(alpha=alpha_ridge[i])
    rr=ridge.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = rr.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
    #print("lambda =", alpha_ridge[i])
    #print("R-squared =", r2)
    #print("Coefficients =", beta)
    

    #Traccio il grafico del modello
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(X_test, y_test,  color='black')
    plt.scatter(X_train, y_train,  color='red')
    plt.plot(X_test, y_pred, color='green', linewidth=3)
    plt.title('Ridge Regression model lambda='+str(alpha_ridge[i]))
    plt.legend(['ridgeRegression','testing set','training set'])
    
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
    textstr = 'MSE='+str(mse)+' \n Beta='+str(beta)   
    plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)
  


# In[36]:


# Fit Lasso Regression (L1)
alpha_lasso = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_lasso)):
    #Costruisco ridge regression sul train-set
    lasso = lm.Lasso(alpha=alpha_lasso[i])
    lar=lasso.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = lar.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
    #print("lambda =", alpha_lasso[i])
    #print("R-squared =", metrics.r2_score(y_test, y_pred))
    #print("Coefficients =", lar.coef_)
    #print("Intercept =", lar.intercept_)

    #Traccio il grafico del modello
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(X_test, y_test,  color='black')
    plt.scatter(X_train, y_train,  color='red')
    plt.plot(X_test, y_pred, color='yellow', linewidth=3)
    plt.title('Lasso Regression model lambda='+str(alpha_lasso[i]))
    plt.legend(['lassoRegression','testing set','training set'])
    
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
    textstr = 'MSE='+str(mse)+' \n Beta='+str(beta)   
    plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)
    


# In[ ]:


# Fit Lasso Regression (L1)
alpha_en = [1e-4, 1e-3, 1e-2,1e-1, 1, 1e2, 1e3]
for i in range(0,len(alpha_lasso)):
    #Costruisco ridge regression sul train-set
    en = lm.ElasticNet(alpha=alpha_en[i], l1_ratio=0.5)
    lar=en.fit(X_train, y_train)

    #Testo la ridge regression sul test-set
    y_pred = lar.predict(X_test)
    mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
    beta=np.round(ridge.coef_,2)
   

    #Traccio il grafico del modello
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(X_test, y_test,  color='black')
    plt.scatter(X_train, y_train,  color='red')
    plt.plot(X_test, y_pred, color='pink', linewidth=3)
    plt.title('Elastic Net model alpha='+str(alpha_en[i])+' l1 ratio=.5')
    plt.legend(['lassoRegression','testing set','training set'])
    
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
    textstr = 'MSE='+str(mse)+' \n Beta='+str(beta)   
    plt.text(0, 0, textstr, fontsize=14, verticalalignment='top', bbox=props)


# In[ ]:

#Costruisco regressione lineare sul train-set
regr = lm.Lars()
lr=regr.fit(X_train, y_train)

#Testo la regressione lineare sul test-set
y_pred = lr.predict(X_test)
mse=np.round(metrics.mean_squared_error(y_test, y_pred),3)
beta=np.round(regr.coef_,2)



#Traccio il grafico del modello
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_test, y_test,  color='black')
plt.scatter(X_train, y_train,  color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Least Angle Regression')
plt.legend(['linear model','testing set','training set'])

#evidenzio gli assi
plt.grid()
plt.axhline(linewidth=2, color='r') #evidenzio asse Ox
plt.axvline(linewidth=2, color='r') #evidenzio asse Oy

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='ivory', alpha=0.9)
textstr = 'MSE='+str(mse)+' \n Beta='+str(beta)   
plt.text(0.1, -1, textstr, fontsize=14, verticalalignment='top', bbox=props)










