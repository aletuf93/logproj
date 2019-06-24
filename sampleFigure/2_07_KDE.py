# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:56:30 2019

@author: Alessandro
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

nPoints=500
x1=np.random.normal(1,1, size=nPoints)
x2=np.random.normal(4,1, size=nPoints)
y=list(x1)+list(x2)

fig, axs = plt.subplots(2, 2)

fig.suptitle('Binning of an empirical distribution')
nbin=7
axs[0,0].hist(y,bins=nbin,color='skyblue')
axs[0,0].set_title('Num bins:'+str(nbin))

nbin=100
axs[0,1].hist(y,bins=nbin,color='skyblue')
axs[0,1].set_title('Num bins:'+str(nbin))

nbin=250
axs[1,0].hist(y,bins=nbin,color='skyblue')
axs[1,0].set_title('Num bins:'+str(nbin))

nbin=500
axs[1,1].hist(y,bins=nbin,color='skyblue')
axs[1,1].set_title('Num bins:'+str(nbin))






def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)
x=y
min_v=-4.5
max_v=10
x_grid = np.linspace(min_v, max_v, 1000)

fig, ax = plt.subplots(2, 2)
band=0.02
pdf = kde_statsmodels_u(x, x_grid, bandwidth=band)
ax[0,0].plot(x_grid, pdf, color='orange', alpha=1, lw=3)
ax[0,0].set_title('bandwith= '+str(band))
ax[0,0].set_xlim(min_v, max_v)

band=0.5
pdf = kde_statsmodels_u(x, x_grid, bandwidth=band)
ax[0,1].plot(x_grid, pdf, color='orange', alpha=1, lw=3)
ax[0,1].set_title('bandwith= '+str(band))
ax[0,1].set_xlim(min_v, max_v)

band=0.7
pdf = kde_statsmodels_u(x, x_grid, bandwidth=band)
ax[1,0].plot(x_grid, pdf, color='orange', alpha=1, lw=3)
ax[1,0].set_title('bandwith= '+str(band))
ax[1,0].set_xlim(min_v, max_v)

band=1
pdf = kde_statsmodels_u(x, x_grid, bandwidth=band)
ax[1,1].plot(x_grid, pdf, color='orange', alpha=1, lw=3)
ax[1,1].set_title('bandwith= '+str(band))
ax[1,1].set_xlim(min_v, max_v)
    
    