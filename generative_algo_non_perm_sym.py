import netket as nk
import openfermion as of
from openfermion import jordan_wigner
from openfermion import get_sparse_operator, get_ground_state
from openfermion import QubitOperator
import jax
import jax.numpy as jnp
from jax import random
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from functools import partial
import optax
from optax import adabelief, sgd
import time

import config

from forging_functions import *
from generative_algo_functions import *




#########    The Hamitonian and the ARNN should be defined in the config file    #########


print("Number of qubits: ", config.n_qubits)
print("Subsystems size: ", config.N)   

H = get_sparse_operator(config.H_of)
E = get_ground_state(H)[0]
print("GS energy:", E)


print('The classical NN defined is a:')
print(config.model)


# Initialize the ARNN
key = random.PRNGKey(123)
key, subkey = random.split(key)
if config.perm_sym == True:
  s = jnp.ones(shape = (8, config.N), dtype=jnp.int32)
else:
  s = jnp.ones(shape = (8, 2*config.N), dtype=jnp.int32)
_, subkey = random.split(subkey)
NN_params = config.model.init(subkey, s)






#########    ARNN Loss    #########


@jax.jit
def kernel(x,y):
  sigma = 1.0
  return jnp.prod(jnp.exp(-1*(x-y)**2/(2*sigma)))
  
@jax.jit
def Loss_ARNN(NN_params, set_bitstring_syst, lambdas):
  """
  loss of the ARNN, MMD loss
  NN_params: parameters of the ARNN
  set_bitstring_syst: set of bitstrings we want to train on
  lambdas: target Schmidt coefficient
  """
  pi = jnp.exp(config.model.apply(NN_params, 2*(A_new-0.5)))**2

  ## MMD
  L = 0
  set_bitstrings = set_bitstring_syst 
  lambdas_squared = lambdas**2

  for i in range(jnp.shape(pi)[0]):
    for j in range(jnp.shape(pi)[0]):
      L += lambdas_squared[i]*lambdas_squared[j]*kernel(set_bitstrings[i],set_bitstrings[j])
      L += pi[i]*pi[j]*kernel(set_bitstrings[i],set_bitstrings[j])
      L += -2*lambdas_squared[i]*pi[j]*kernel(set_bitstrings[i],set_bitstrings[j])
  return L

val_grad_ARNN = jax.jit(jax.value_and_grad(Loss_ARNN, argnums = 0))




#########    Simulation    #########


# Initialize the sets of bitstrings
A = jnp.array([[0, 1, 0, 0, 0, 0],
       [1, 0, 1, 0, 0, 1],
       [0, 0, 0, 0, 1, 1],
       [1, 1, 1, 1, 0, 1],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0, 1]], dtype = np.int32)
B = jnp.array([[0, 1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 1],
       [1, 1, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 0, 1],
       [1, 1, 1, 1, 1, 0],
       [0, 1, 0, 1, 1, 0]], dtype = np.int32)



lr = 0.001
optARNN = adabelief(learning_rate=lr) 
opt_stateARNN = optARNN.init(NN_params)

swap_A = []
swap_B = []
history_loss = []

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


start = time.time()
print("start jitting the generative loop")
A_new, B_new, set_bitstring_syst_new, lambdas_new = Generative_loop_non_perm_sym(NN_params, params_A, params_B, A, B, subkey)
end = time.time()
jit_time = (end-start)/60
print("Time to Jit: ", jit_time, " Minutes")


print("We start with A0: ")
print(A)
print("and B0: ")
print(B)

nbr_iter_loop_gene = 100
nbr_iter_ARNN = 1

for i in range(nbr_iter_loop_gene):
  _, subkey = random.split(subkey)
  A_new, B_new, set_bitstring_syst_new, lambdas_new = Generative_loop_non_perm_sym(NN_params, params_A, params_B, A, B, subkey)

  kkA = k-count_common_rows(A_new, A)
  kkB = k-count_common_rows(B_new, B)

  swap_A.append(kkA.tolist())
  swap_B.append(kkB.tolist())
  A = A_new
  B = B_new
  print("-----------------   This is the generative loop numbre {}   -----------------".format(i))
  print("{} bitstring(s) has(ve) been added to A, the new set is: ".format(kkA))
  print(A_new)
  print("{} bitstring(s) has(ve) been added to B, the new set is: ".format(kkB))
  print(B_new)


  print("Optimization of the ARNN:")
  for j in range(nbr_iter_ARNN):
    val, grad = val_grad_ARNN(NN_params, set_bitstring_syst_new, lambdas_new)
    updatesARNN, opt_stateARNN = optARNN.update(grad, opt_stateARNN)
    NN_params = optax.apply_updates(NN_params, updatesARNN)
    print("step: {}, loss: {}".format(j,val))
  history_loss.append(val.tolist())




print(history_loss)
print(swap_A)
print(swap_B)
print(A)
print(B)
  
