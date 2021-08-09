# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:16:43 2021

@author: atteb
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics

spots = []
options = []
mc = []

for spot in range(70,135,5):
        
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
    
    train_size = 50000
    
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
    
    
    output_shape = payoffs[0].shape
    
    nodes_per_layer = 32
    N_hidden_layers = 3
    hidden_activation = keras.activations.relu
    output_activation = keras.activations.linear
    
    # TD MODEL
    td_model = TD_net(
    
        #hidden layers
        nodes_per_layer = nodes_per_layer,
        N_hidden_layers = N_hidden_layers,
        hidden_activation = hidden_activation,
       
        #output
        output_shape = output_shape,            
        output_activation = output_activation,
        row_wise_output = False
    )
    
    td_model.compile(
        optimizer = tf.keras.optimizers.SGD(),
        metrics = [td_metrics.Temporal_MAE, td_metrics.Prediction_MAE]
    )
            
    td_hist = td_model.fit(
       
    # training data
    walks_with_time[:2000],
    payoffs[:2000],
       
    # hyperparameters
    lr = 0.001,
    batch_size=2,
    lamb = 0.3,
    validation_split=0,
    epochs = 10,
    verbose = True,
    #steps_per_epoch = 1000
    )
       
    t_0_pred = td_model( walks_with_time[0] )[0].numpy()
    
    options.append(t_0_pred)
    spots.append(spot)    
    mc.append(montecarlo)
    
    #rmse = np.sqrt(sum((montecarlo - t_0_pred)**2)/2000)
    
    

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Spot Price")
ax.set_ylabel("Option Price")
ax.title.set_text("RMSE of TD Model - Varying number of inputs")

ax.plot(spots,options, linestyle='--', marker='o',label = "TD")
ax.plot(spots,mc, linestyle='--', marker='o',label = "MC")
ax.legend()



