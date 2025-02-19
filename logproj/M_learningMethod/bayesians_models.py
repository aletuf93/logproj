# -*- coding: utf-8 -*-


# import math functions
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB


tuned_param_nb= [{'var_smoothing':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-2,1e-1,1,1e2,1e3]}]
                                


models_classification = {
                            'naive bayes': {
                                   'estimator': GaussianNB(), 
                                   'param': tuned_param_nb,
                            },
                        }







# %% KALMAN FILTER METHODS

#  Kalman filter for compensation between the cynematic and the model
#https://towardsdatascience.com/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968

def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]

# measurements for mu and motions, U
#measurements = [5., 6., 7., 9., 10.]
#motions = [1., 1., 2., 1., 1.]

# initial parameters
#measurement_sig = 4.
#motion_sig = 2.



def OneDimensionalKalmanFilter(measurements_prior, motions_posterior):
    #measurement is a series of measurements of the hidden state (e.g. the inventory) given by a model or a sensor
    #if the state is observable. This is the prior
    
    #motions is a series of value of transitions between states (e.g. movements changing the inventory). This is the posterior
    
    mu = 0.
    sig = 10000.
    
    prior_sig=np.std(measurements_prior)
    posterior_sig=np.std(motions_posterior)
    
    print(f"**Kalman filter running unsing***")
    print(f"**prior sigma: {np.round(prior_sig,2)}")
    print(f"**posterior sigma  prior sigma: {np.round(posterior_sig,2)}")
    
    if (posterior_sig>prior_sig):
        print("*****WARNING, sigma posterior > sigma prior *********")
        print("****Kalman filted cannot improve uncertainty *********")
    
    #measure_mean_update=[]
    #measure_sigma_update=[]
    hidden_state_mean_predict=[]
    hidden_state_sigma_predict=[]
    for n in range(len(measurements_prior)):
        
        # measurement update, with uncertainty
        mu, sig = update(mu, sig, measurements_prior[n], prior_sig)
        #measure_mean_update.append(mu)
        #measure_sigma_update.append(sig)
        #print('Update: [{}, {}]'.format(mu, sig))
        
        # motion update, with uncertainty
        mu, sig = predict(mu, sig, motions_posterior[n], posterior_sig)
        hidden_state_mean_predict.append(mu)
        hidden_state_sigma_predict.append(sig)
        #print('Predict: [{}, {}]'.format(mu, sig))
    
        
    # print the final, resultant mu, sig
    print('\n')
    print('Final result: [{}, {}]'.format(mu, sig))
    return hidden_state_mean_predict, hidden_state_sigma_predict



    
