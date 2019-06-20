# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:24:22 2019

@author: Alessandro
"""
import numpy as np
import matplotlib.pyplot as plt

N=600
x=np.linspace(0,1,N)
y=np.sin(50*2*np.pi*x)+0.5*np.sin(80*2*np.pi*x)
plt.plot(x,y,color='skyblue',marker='o')
