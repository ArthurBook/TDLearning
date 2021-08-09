import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from StochasticProcesses import Stochastic_Processes
from Payoffs import Payoffs

from TD_net_v4 import TD_net, td_metrics


#%%
process_specs = {
        "processdicts" : [
                {"process" : "GBM", "s0" : 100, 'mu': 0, 'v': 0.01},
        ],
        "d_t" : 3 / 360,
        "N_steps" : 50,
        "batch_size" : 1000,
}

train_size = 1000

walks = Stochastic_Processes.generate_walks( **process_specs, N_walks = train_size, verbose =False )

s = walks.shape
timedim = np.tile( range( s[1] ), s[0] ).reshape(*s[0:-1],1)
walks_with_time = np.concatenate( [walks,timedim], axis = -1 )

payoffs = Payoffs.european_option(
    walks,
    100,
    "call"
    )
    
montecarlo = np.mean( payoffs, 0 )

#%%

output_shape = payoffs[0].shape

nodes_per_layer = 32
N_hidden_layers = 0
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
lrs = [0.00001,0.0001,0.001,0.01,0.1,1]

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Lambda")
ax.set_ylabel("RMSE")
ax.title.set_text("RMSE of TD as Lambda changes - Hidden layers = 0, TD Inputs = 500")


for j in lrs:

    lambdas = []
    rmses = []
    
    for i in range(21):
        
        lamb = i/20
        
        td_hist = td_model.fit(
        
        walks_with_time[:500],
        payoffs[:500],
        
        lr = j ,
        batch_size=2,
        lamb = lamb,
        validation_split=0,
        epochs = 10,
        verbose = True,
        )
        
        t_0_pred = td_model( walks_with_time[0] )[0].numpy()
               
        rmse = np.sqrt(sum((montecarlo - t_0_pred)**2)/500)
        
        lambdas.append(lamb)
        rmses.append(rmse)
    
    ax.plot(lambdas,rmses, linestyle='--', marker='o',label = f"lr = {j}")
    
ax.legend()