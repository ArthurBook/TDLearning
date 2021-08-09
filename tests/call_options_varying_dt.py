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




#%%

list_rmse = []
#dt_list = [1/48,1/24,1/16,1/12,5/48,1/8]
dt_list = [1/8,4/8,1,12/8,2,20/8,3]

for dt in dt_list:
    
    process_specs = {
        "processdicts" : [
                {"process" : "Bates", "s0" : 100, 'mu': 0.05, 'v0': 0.04, 'rho' : -0.2, 'theta': 0.04, 'xi': 0.04, 'kappa': 0.3, 'lamb': 0.75, 'k': 0.0625,
                 #"randomize_spot" : [-20,20]
                 },
        ],
        "d_t" : dt / 360,
        "N_steps" : 10,
        "batch_size" : 1000,
        }

    train_size = 50000
    
    walks = Stochastic_Processes.generate_walks( **process_specs, N_walks = train_size, verbose =False )
    
    #Add time dim to the walks
    s = walks.shape
    
    timedim = np.tile( range( s[1] ), s[0] ).reshape(*s[0:-1],1)
    walks_with_time = np.concatenate( [walks,timedim], axis = -1 )
    
    payoffs = Payoffs.discrete_mass(
        walks,
        space = np.linspace(80,120,21)
    )
    
    
    montecarlo = np.mean( payoffs, 0 )
    
    output_shape = payoffs[0].shape
    
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

    
    td_hist = td_model.fit(
       
        # training data
        walks_with_time[:500],
        payoffs[:500],
       
        # hyperparameters
        lr = 0.0001,
        batch_size=2,
        lamb = 0.275,
        validation_split=0,
        epochs = 10,
        verbose = True,
    )
       
    t_0_pred = td_model( walks_with_time[0] )[0].numpy()
           
    rmse = np.sqrt(sum((montecarlo - t_0_pred)**2)/500)
   
    list_rmse.append(rmse)
    
fig5, ax5 = plt.subplots(1,1)
ax5.set_xlabel("dt")
ax5.set_ylabel("RMSE")
ax5.title.set_text("RMSE of TD as dt changes")

ax5.plot(dt_list,list_rmse, linestyle='--', marker='o')
    
