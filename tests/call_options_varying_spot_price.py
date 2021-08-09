import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics


#%%

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Spot Price")
ax.set_ylabel("Option Price")
ax.title.set_text("Montecarlos")

mc1_spots = []
mc1 = []

mc2 = []


for spot in range(81,121,2):
        
    process_specs = {
            "processdicts" : [
                    {"process" : "Bates", "s0" : spot, 'mu': 0.05, 'v0': 0.04, 'rho' : -0.2, 'theta': 0.04, 'xi': 0.04, 'kappa': 0.3, 'lamb': 0.75, 'k': 0.0625,
                     #"randomize_spot" : [-20,20]
                     },
            ],
            "d_t" : 3 / 360,
            "N_steps" : 10,
            "batch_size" : 1000,
    }
    
    train_size = 5000000
    
    walks = Stochastic_Processes.generate_walks( **process_specs, N_walks = train_size, verbose =False )
    
    #Add time dim to the walks
    s = walks.shape
    
    timedim = np.tile( range( s[1] ), s[0] ).reshape(*s[0:-1],1)
    walks_with_time = np.concatenate( [walks,timedim], axis = -1 )
    
    payoffs = Payoffs.european_option(
    walks,
    100,
    "call"
    )
    
    montecarlo = np.mean( payoffs, 0 )
    
    mc1_spots.append(spot)    
    mc1.append(montecarlo)
    
#%%
 
process_specs = {
        "processdicts" : [
                {"process" : "Bates", "s0" : 100, 'mu': 0.05, 'v0': 0.04, 'rho' : -0.2, 'theta': 0.04, 'xi': 0.04, 'kappa': 0.3, 'lamb': 0.75, 'k': 0.0625, 
                 "randomize_spot" : [-20,20] 
                 },
        ],
        "d_t" : 3 / 360, 
        "N_steps" : 10,
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

for t in discretized_time:
    
    walks_t = walks[:,t,0]  
    
    for lb,ub,mid in zip(lower_bounds, upper_bounds, bin_centers):
        
        in_discrete_bin = ( lb <= walks_t ) * ( walks_t <= ub ) 
        
        subset_of_payoffs_with_condition = payoffs[in_discrete_bin] 
        
        conditional_expectation.loc[t,mid] = np.mean( subset_of_payoffs_with_condition )

mc2 = []
mc2_spots = [i for i in range(81,121,2)]   

for spot in mc2_spots:
    
    mc = conditional_expectation[spot][0]
    mc2.append(mc)
    
   
    
    
ax.plot(mc2_spots,mc2,linestyle='--', marker='o',label = "MC2")
ax.plot(mc1_spots,mc1, linestyle='--', marker='o',label = "MC1")
ax.legend()    
    