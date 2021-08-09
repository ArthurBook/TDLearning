import numpy as np
import pandas as pd
import math

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

df = pd.read_excel("variables.xlsx").astype(float)

                                        # Loop 8 times and do average                
ns = [1,5,10,25,50,75, 100]
lrs = [0.0000316227766016837,0.0001,0.000316227766016837,0.001,0.00316227766016837,0.01]
lambdas = [i/10 for i in range(0,11)]
    
for lamb in lambdas:
    
    print(f"lambda = {lamb}")
    
    dfs =[]
    
    for n in ns:
        
        print(f"n = {n}")
                        
        for lr in lrs:
            
            print(f"lr = {lr}")
            
            row_index = df.index[abs(df["temp"] - lr) < 0.0000001][0]
            
            process_specs = {
            "processdicts" : [
                    {"process" : "Bates", "s0" : 100, 'mu': 0.05, 'v0': 0.04, 'rho' : -0.2, 'theta': 0.04, 'xi': 0.04, 'kappa': 0.3, 'lamb': 0.75, 'k': 0.0625,
                     #"randomize_spot" : [-20,20]
                     },
            ],
            "d_t" : (1/8) / 360,
            "N_steps" : n,
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
            walks_with_time[:500],
            payoffs[:500],
               
            # hyperparameters
            lr = lr,
            batch_size=2,
            lamb = lamb,
            validation_split=0,
            epochs = 10,
            verbose = True,
            #steps_per_epoch = 1000
            )
               
            t_0_pred = td_model( walks_with_time[0] )[0].numpy()
    
            if math.isnan(t_0_pred):
                
                df[n][row_index] = 0
                
            else:
                
                df[n][row_index] = 1
    
    name = f'C:/Users/alexa/Desktop/Stonks/variables{str(round(lamb,8))}.csv' 

    df.to_csv(name,index = False)


    

   





