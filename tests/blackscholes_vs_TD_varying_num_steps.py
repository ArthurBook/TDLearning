import py_vollib as pv
from py_vollib.ref_python import black_scholes
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics




dt = 1/360
n_steps = [10,50,100]
#from time 0, 3 curves 1 for each n, mc2 vs blackscholes

flag = 'c'
S = 100
K = 100
dt = 1/360
r = 0.1
sigma = 0.2
var = sigma**2

for n_step in n_steps:
     
    fig2, ax2 = plt.subplots(1,1)
    
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Error")
    ax2.title.set_text(f"Error between Montecarlo and Black-Scholes, N_steps = {n_step}")
    
    fig1, ax1 = plt.subplots(1,1)

    ax1.set_xlabel(f"Time Step (out of {n_step})")
    ax1.set_ylabel("RMSE")
    ax1.title.set_text(f"RMSE BlackScholes Vs Montecarlo, N_steps = {n_step}")
    
    print(f"n_step = {n_step}")
    
    T = n_step * dt

    if n_step == 10:
        
        ns = [i for i in range(0,12,2)]
        
    elif n_step == 50:
        
        ns = [i for i in range(0,60,10)]
        
    else:
        
        ns = [i for i in range(0,120,20)]
        
    
                
    process_specs = {
        "processdicts" : [
                {"process" : "GBM", "s0" : S, 'mu': r, 'v': var,
                 "randomize_spot" : [-20,20]},
        ],
        "d_t" : dt,
        "N_steps" : n_step,
        "batch_size" : 1000,
    }
    
    sample_size = 5000000

    discretized_space = np.linspace(80,120,21) 
    discretized_time = range( 0, process_specs["N_steps"] + 1, 1 )
    
    lower_bounds = discretized_space[:-1]
    upper_bounds = discretized_space[1:]
    bin_centers = ( lower_bounds + upper_bounds ) / 2 
    
    walks = Stochastic_Processes.generate_walks( **process_specs, N_walks = sample_size, verbose =False )

    payoffs = np.maximum( walks[:,-1,0] - 100, 0 )

    conditional_expectation = pd.DataFrame() 

    for temp in discretized_time:
        
        walks_t = walks[:,temp,0]  
        
        for lb,ub,mid in zip(lower_bounds, upper_bounds, bin_centers):
            
            in_discrete_bin = ( lb <= walks_t ) * ( walks_t <= ub ) 
            
            subset_of_payoffs_with_condition = payoffs[in_discrete_bin] 
            
            conditional_expectation.loc[temp,mid] = np.mean( subset_of_payoffs_with_condition )
        
    rmse_list = []
    
    for N_prime in ns:
        
        fig, ax = plt.subplots(1,1)

        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Expected Payoff")
        ax.title.set_text(f"BlackScholes Vs Montecarlo, timestep = {N_prime} / {n_step}")
        
        t = N_prime*dt
        N_dub_prime = n_step - N_prime
        time = N_dub_prime*dt   
        
        mc2_list = []
        bs_list = []
        x_axis = []
        error_list = []
        for space in range(81,121,2):
            
            print(f"spot = {space}")
            
            x_axis.append(space)
            
            mc2 = conditional_expectation[space][N_prime]
            
            bs = black_scholes.black_scholes(flag,space,K,time,r,sigma)
            
            error = np.sqrt((mc2-bs)**2)
            
            mc2_list.append(mc2)
            bs_list.append(bs)
            error_list.append(error)
            
        rmse = sum(error_list) / len(error_list)
        rmse_list.append(rmse)
                        
        ax.plot(x_axis,mc2_list,label = f"mc2, time_step = {N_prime}",linestyle='--', marker='o')
        ax.plot(x_axis,bs_list,label = f"bs, time_step = {N_prime}",linestyle='--', marker='o')
        ax2.plot(x_axis,error_list,label = f"time_step = {N_prime}",linestyle='--', marker='o')
        ax.legend()
        ax2.legend()
    
    ax1.plot(ns,rmse_list,linestyle='--', marker='o')

        
    ax.legend()
    ax2.legend()
    

        
        


