# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:04:59 2019

@author: Alessandro
"""
import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

#random data
data = np.random.normal(0,5, size=200)

#empirical CDF
ecdf = ECDF(data)

#fit data to distribution
mu, std = norm.fit(data)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Empirical and best-fit PDF and CDF')

#plot empirical

axs[0,0].hist(data, color='skyblue', bins=len(data))
axs[0,0].set_title('Histogram - PDF')
axs[0,1].plot(ecdf.x,ecdf.y, color='skyblue')
axs[0,1].set_title('Empirical - CDF')

#plot fitted

# Plot the histogram
n, bins, patches =axs[1,0].hist(data, bins=25, density=True, alpha=0.6, color='skyblue')
# Plot the PDF.
xmin, xmax = axs[1,0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
axs[1,0].plot(x, p, 'orange', linewidth=2)
title = "Gaussian Fit PDF: mu = %.2f,  std = %.2f" % (mu, std)
axs[1,0].set_title(title)


# plot the CDF
# Add a line showing the expected distribution.
y = ((1 / (np.sqrt(2 * np.pi) * std)) *
     np.exp(-0.5 * (1 / std * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]

axs[1,1].plot(bins, y, 'orange', linewidth=1.5,)
title = "Gaussian Fit CDF: mu = %.2f,  std = %.2f" % (mu, std)
axs[1,1].set_title(title)





