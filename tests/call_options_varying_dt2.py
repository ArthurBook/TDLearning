import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics

#%%

#output_shape = payoffs[0].shape
output_shape = ()

nodes_per_layer = 64
N_hidden_layers = 5
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

#%%

def input_creation(strike, n_steps,specs, sample_size, td_input_count, central_weight ):
    
    walks = Stochastic_Processes.generate_walks( **specs, N_walks = sample_size, verbose =False )
    
    walks_remove_dimension = walks[:,:,0]
    df_walks = pd.DataFrame(walks_remove_dimension)
    
    df_walks_sorted = df_walks.sort_values(by =n_steps,ignore_index=True)
    np_sorted_walks = df_walks_sorted.to_numpy()
    walks = np_sorted_walks[..., np.newaxis]

    s = walks.shape
    timedim = np.tile( range( s[1] ), s[0] ).reshape(*s[0:-1],1)
    walks_with_time = np.concatenate( [walks,timedim], axis = -1 )
    payoffs = np.maximum( walks[:,-1,0] - 100, 0 ) 
    
    central_td_inputs_count = int(central_weight * td_input_count)
    central_indexs = df_walks_sorted.index[abs(df_walks_sorted[n_steps] - strike) < 0.0001].tolist()
    central_index = central_indexs[int(len(central_indexs)/2)]
    lowest_central_index = central_index - int(central_td_inputs_count/2)
    highest_central_index = central_index + int(central_td_inputs_count/2)
    central_walks_td = walks_with_time[lowest_central_index : highest_central_index]
    central_payoffs_td = payoffs[lowest_central_index : highest_central_index]

    sample_size = len(walks)
    smallest = df_walks_sorted[n_steps][0]
    largest = df_walks_sorted[n_steps][sample_size-1]
    expected_range = largest - smallest
    
    walks_remaining = td_input_count - len(central_walks_td)
    
    numb_walks_left_sides = int(walks_remaining / 2)     # number of walks we can generate on the left / right
    numb_walks_right_sides = int(walks_remaining / 2)
    
    
    left_buckets = int(lowest_central_index / numb_walks_left_sides)
    right_buckets = int(((sample_size-1) - highest_central_index) / numb_walks_right_sides)
    
    indexs_left = [i for i in range(0,lowest_central_index,left_buckets)]
    indexs_right = [i for i in range(highest_central_index,sample_size-1,right_buckets)]
    
    side_indexs = indexs_left + indexs_right
    
    side_walks_td = (walks_with_time[side_indexs])
    side_payoffs_td = (payoffs[side_indexs])
    
    walks_td = np.vstack((central_walks_td, side_walks_td))
    payoffs_td = np.hstack((central_payoffs_td, side_payoffs_td))
    
    return walks_td,payoffs_td


#%%

fig2, ax2 = plt.subplots(1,1)

ax2.set_xlabel("Number of Steps")
ax2.set_ylabel("RMSE")
ax2.title.set_text("RMSE against number of steps - varying times")

dts = [1/8,4/8,1,12/8,2,20/8,3]

for time in range(0,12,2):
    
    rmses = []
    dt_list = []
        
    for dt in dts:
        
        process_specs = {
        "processdicts" : [
                {"process" : "GBM", "s0" : 100, 'mu': 0, 'v': 0.01, 
                 "randomize_spot" : [-20,20] 
                 },
        ],
        "d_t" : dt / 360, 
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
                
        
        dt_list.append(dt)
        
        walks_td,payoffs_td = input_creation(100, 10, process_specs, 20000000, 500, 0.333)
    
        td_hist = td_model.fit( 
    
        walks_td,
        payoffs_td,
        
        lr = 0.00001 ,
        batch_size=2,
        lamb = 0.3,
        validation_split=0,
        epochs = 30,
        verbose = True,
        )
        
    
        errors = []
    
        for space in range(81,121,2):
            
            t_pred = float(td_model.predict([[space,time]]))
            montecarlo = conditional_expectation[space][time]
            error = np.sqrt((montecarlo-t_pred)**2)
            errors.append(error)
            
        rmse = sum(errors)/len(errors)
        rmses.append(rmse)
        
    ax2.plot(dt_list,rmses,label = f"time = {time}",linestyle='--', marker='o')
    
ax2.legend()