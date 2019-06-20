# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:24:22 2019

@author: Alessandro
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

N=600
T = 1.0 / 800.0
x=np.linspace(0,N*T,N)
y=np.sin(50*2*np.pi*x)+0.5*np.sin(80*2*np.pi*x)


#plot samples (Figure 19)
plt.plot(x,y,color='skyblue',marker='o')
plt.title('Samples')

#Fourier Transform

#plot fft (Figure 20)
plt.figure()
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),color='skyblue')
plt.title('Amplitude spectrum')
