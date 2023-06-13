import openfermion as of
from openfermion import get_sparse_operator, get_ground_state
from openfermion import QubitOperator
import jax
import jax.numpy as jnp
from jax import random
import pennylane as qml
import optax
from optax import adabelief, sgd
import time

import config

from forging_functions import *

#########    Define the of Hamitonian    #########


print("Number of qubits: ", config.n_qubits)
print("Subsystems size: ", config.N)   

H = get_sparse_operator(config.H_of)
E = get_ground_state(H)[0]
print("GS energy:", E)



#########    Simulation    #########


bitstringA = jnp.array([[1, 0, 0, 1, 1, 0],
       [1, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0],
       [1, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 1, 0],
       [1, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 1]], dtype=jnp.int32)

bitstringB = jnp.array([[1, 0, 0, 1, 0, 0],
       [0, 1, 0, 1, 0, 1],
       [0, 1, 0, 0, 0, 1],
       [0, 0, 0, 1, 1, 1],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 1, 1],
       [0, 0, 1, 1, 0, 1]], dtype=jnp.int32)


n_layers = 15
Schmidt_rank = 8

#for QC
k = 8 #cutoff

n_layers = 5 #number of layers in the variational circuits
params_shape = (n_layers, config.N, 3)
key = random.PRNGKey(1234)
key, subkey = random.split(key)
params_A = random.uniform(subkey, params_shape, dtype = np.float32) #parameters for the circuit acting on subsystem A
key, subkey = random.split(key)
params_B = random.uniform(subkey, params_shape, dtype = np.float32) #paramters for the circuit acting on subsystem B
Schmidt_coef = jnp.ones((k,), dtype = np.float32) #Schmidt coefficient
Schmidt_coef = Schmidt_coef/jnp.sqrt(jnp.sum(Schmidt_coef**2))

lr = 0.01
epochs = 4000

optA = adabelief(learning_rate=lr)
opt_stateA = optA.init(params_A)
optB = adabelief(learning_rate=lr)
opt_stateB = optB.init(params_B)
optS = sgd(learning_rate=lr, momentum=0.6, nesterov=True)
opt_stateS = optS.init(Schmidt_coef)

measure_progress = []
paramsA_progress = []
paramsB_progress = []
schmidt_progress = []
loss_progress = []


start = time.time()
print("#####   Start jitting the gradients   #####")
measure, grads_E_A = grad_E_fn_circA(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
print("grad circuit A jitted")
measure, grads_E_B = grad_E_fn_circB(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
print("grad circuit B jitted")
measure, grads_E_S = grad_E_fn_schmidt(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
print("grad schmidt coef jitted")
end = time.time()
jit_time = (end-start)/60
print("Time total to Jit: ", jit_time, " Minutes")

name_models = "TFIM_1d_12s_rdmh"

print("#####   Starting the VQE   #####")

for i in range(epochs):
    _, subkey = random.split(subkey)
    
    measure, grads_E_A = grad_E_fn_circA(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
    measure, grads_E_B = grad_E_fn_circB(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
    updatesA, opt_stateA = optA.update(grads_E_A, opt_stateA)
    params_A = optax.apply_updates(params_A, updatesA)
    updatesB, opt_stateB = optB.update(grads_E_B, opt_stateB)
    params_B = optax.apply_updates(params_B, updatesB)

    if i%10==0:
      measure, grads_E_S = grad_E_fn_schmidt(params_A, params_B, Schmidt_coef, bitstringA,bitstringB)
      updatesS, opt_stateS = optS.update(grads_E_S, opt_stateS)
      Schmidt_coef = optax.apply_updates(Schmidt_coef, updatesS)
      Schmidt_coef = Schmidt_coef/jnp.sqrt(jnp.sum(Schmidt_coef**2))

      E = measure
      measure_progress.append(E.tolist())
      paramsA_progress.append(params_A.tolist())
      paramsB_progress.append(params_B.tolist())
      schmidt_progress.append(Schmidt_coef.tolist())
      print('Loss step {}: {}'.format(i, E))

import json
with open('history_loss_{}_rdmset.txt'.format(name_models), 'w') as f:
    f.write(json.dumps(measure_progress))
with open('Schmidt_coef_{}_rdmset.txt'.format(name_models), 'w') as f:
    f.write(json.dumps(schmidt_progress))
with open('ParamsA_{}_rdmset.txt'.format(name_models), 'w') as f:
    f.write(json.dumps(paramsA_progress))
with open('ParamsB_{}_rdmset.txt'.format(name_models), 'w') as f:
    f.write(json.dumps(paramsB_progress))
