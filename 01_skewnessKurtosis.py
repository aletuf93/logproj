# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:46:07 2019

@author: Alessandro
"""

import numpy as np
from scipy.stats import norm, lognorm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

# shape factor of the distribution
s1 = 1
s2 = 0.5
s3 = 0.25
s4=0.1
s5=0.01
s6=0.001

mean1, var1, skew1, kurt1 = lognorm.stats(s1, moments='mvsk')
x1 = np.linspace(lognorm.ppf(0.01, s1),lognorm.ppf(0.99, s1), 100)

mean2, var2, skew2, kurt2 = lognorm.stats(s2, moments='mvsk')
x2 = np.linspace(lognorm.ppf(0.01, s2),lognorm.ppf(0.99, s2), 100)

mean3, var3, skew3, kurt3 = lognorm.stats(s3, moments='mvsk')
x3 = np.linspace(lognorm.ppf(0.01, s3),lognorm.ppf(0.99, s3), 100)

mean4, var4, skew4, kurt4 = lognorm.stats(s4, moments='mvsk')
x4 = np.linspace(lognorm.ppf(0.01, s4),lognorm.ppf(0.99, s4), 100)

mean5, var5, skew5, kurt5 = lognorm.stats(s5, moments='mvsk')
x5 = np.linspace(lognorm.ppf(0.01, s5),lognorm.ppf(0.99, s5), 100)

mean6, var6, skew6, kurt6 = lognorm.stats(s6, moments='mvsk')
x6 = np.linspace(lognorm.ppf(0.01, s6),lognorm.ppf(0.99, s6), 100)


fig, axs = plt.subplots(3,2)
axs[0,0].plot(x1, lognorm.pdf(x1, s1), 'orange', linewidth=2)
axs[0,0].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean1, var1, skew1, kurt1))
axs[1,0].plot(x2, lognorm.pdf(x2, s2), 'orange', linewidth=2)
axs[1,0].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean2, var2, skew2, kurt2))
axs[2,0].plot(x3, lognorm.pdf(x3, s3), 'orange', linewidth=2)
axs[2,0].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean3, var3, skew3, kurt3))
axs[0,1].plot(x4, lognorm.pdf(x4, s4), 'orange', linewidth=2)
axs[0,1].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean4, var4, skew4, kurt4 ))
axs[1,1].plot(x5, lognorm.pdf(x5, s5), 'orange', linewidth=2)
axs[1,1].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean5, var5, skew5, kurt5 ))
axs[2,1].plot(x6, lognorm.pdf(x6, s6), 'orange', linewidth=2)
axs[2,1].set_title('Lognormal with mean= %.2f, var= %.2f, skewness= %.2f kurtosis= %.2f' % (mean6, var6, skew6, kurt6 ))

