# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:05:29 2020

@author: aletu
"""

import numpy as np
import matplotlib.pyplot as plt

# %% CREATE PRODUCTION AND DEMAND FUNCTIONS

number_of_sample = 365 #days
mu_production = 100 #units per day
sigma_production = 10 # units per day

mu_demand = 100 #units per day
sigma_demand = 15 # units per day


x = np.random.normal(mu_production,sigma_production,number_of_sample) 
d = np.random.normal(mu_demand,sigma_demand,number_of_sample) 

# represent demand
plt.hist(d,color='orange')
plt.hist(x,color='skyblue')

plt.title('Production and Demand distribution')
plt.xlabel('Daily rate')
plt.ylabel('Frequency')
plt.legend(['Demand','Production'])

x = np.array(x)
d = np.array(d)
# %% DEFINE THE INVENTORY FUNCTION q

q = [0]
for i in range(0,len(d)):
    inventory_value = q[i] + x[i] - d[i] 
    if inventory_value <0 : inventory_value=0
    q.append(inventory_value)
    
plt.plot(q)
plt.xlabel('days')
plt.ylabel('Inventory quantity $q$')
plt.title('Inventory function $q$')

q = np.array(q)

# %% DEFINE THE FUNCTION p
p = [q[i]-q[i-1] for i in range(1,len(q))]
plt.plot(p)
plt.xlabel('days')
plt.ylabel('Function $p$')
plt.title('Value')

p=np.array(p)
    
# %% DEFINE THE LINEAR POTENTIAL V(q)
F0 = 1
V_q = -F0*q
V_q = V_q[0:-1]

# %% DEFINE L(q,q)
L_qq = 1/2*(np.array(p))**2 - V_q

plt.plot(L_qq)
plt.xlabel('days')
plt.ylabel('value')
plt.title('Function $L(q,\dot{q})$')



# %% DEFINE H
H = (p**2)/2 - F0*q[0:-1]

plt.plot(H)
plt.xlabel('days')
plt.ylabel('value')
plt.title('Function $H$')

# %% DEFINE q from H

S_q = [H[i-1] + H[i] for i in range(1,len(H))]
plt.plot(S_q)
plt.xlabel('days')
plt.ylabel('value')
plt.title('Function $S[q]$')



#%% MODEL WIENER PROCESS

"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

# File: brownian.py

from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out





# %% MODEL

#F0 defined above
eta = 0.5 #days to reach the final value of inventory
beta = 15 #diffusion coefficient
Fr_t = brownian(x0=0, n=365, dt=1, delta=beta, out=None) #demand stochastic process
p_dot = F0 -eta*p + beta*Fr_t

# %%

plt.plot(Fr_t)
plt.plot(p_dot)

