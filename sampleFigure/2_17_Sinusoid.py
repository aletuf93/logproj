# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:50:30 2019

@author: Alessandro
"""
# Figure 17
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,20,10000)
y=np.sin(x-3)
plt.plot(x,y,'k')