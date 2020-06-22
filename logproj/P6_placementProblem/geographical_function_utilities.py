# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import griddata

# %% represent a 3D function from points
def surfaceFromPoints(D_input,xCol,yCol,zCol):
    #identifico il rettangolo da rappresentare
    min_x = min(D_input[xCol])
    max_x= max(D_input[xCol])

    min_y=min(D_input[yCol])
    max_y=max(D_input[yCol])

    #costruisco la griglia
    x = np.linspace(min_x,max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X, Y = np.meshgrid(x, y)
    xy_coord = list(zip(D_input[xCol],D_input[yCol]))

    #interpolo la funzione nei punti mancanti
    grid = griddata(xy_coord, np.array(D_input[zCol]), (X, Y), method='linear')

    return X, Y, grid




# %%

''' #plot 3d function with matplotlib
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, grid)
'''
