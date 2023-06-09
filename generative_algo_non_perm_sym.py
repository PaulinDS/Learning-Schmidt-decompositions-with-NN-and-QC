from generative_algo_functions import *
from forging_functions import *

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

from jax.lax import scan
from jax.lax import cond
from jax.lax import dynamic_slice

import optax
from optax import adabelief, noisy_sgd, yogi, adam, sgd
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_l2_sphere



#########    Define the of Hamitonian    #########


#TFIM 1d 14 spins, rdm uni ext field
n_qubits = 14
key = random.PRNGKey(1234)
key, subkey = random.split(key)
h_i = random.uniform(subkey, (n_qubits,), dtype = jnp.float32).tolist()
H_of = QubitOperator('Z{} Z0'.format(n_qubits-1))
  H_of += QubitOperator('Z{} Z{}'.format(i, i+1)) 
for i in range(n_qubits):
  H_of += h_i[0]*QubitOperator('X{}'.format(i))
  
n_qubits = n_qubits
N = n_qubits//2
print("Number of qubits: ", n_qubits)
print("Subsystems size: ", N)   



#########    Convert the of Hamitonian    #########

"""
group the operators into overlapping and local ones, convert them into qml, code taken and adapted from:
https://github.com/cqsl/Entanglement-Forging-with-GNN-models
"""

gates = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "I":qml.Identity}

op_list_list = []

local_operators_A = []
local_operators_B = []
overlap_operators = []

k = 0
for o in H_of: 
    k += 1
    if k!=-2: #to remove constant terms, if any
        for keys in o.terms.keys():
            indices = np.array(keys)[:,0].astype(np.int64)
            Ops = np.array(keys)[:,1]
            op_list = [[np.real(x) for x in o.terms.values()][0]] 
            for i in range(n_qubits//2):
                if np.isin(i, indices):
                    idx = np.where(indices == i)
                    O = gates[Ops[idx][0]](i)
                else:
                    O = gates["I"](i)
                if i == 0:
                    op = O
                else:
                    op = op@O
            op_list.append(op)
            for i in range(n_qubits//2, n_qubits):
                if np.isin(i, indices):
                    idx = np.where(indices == i)
                    O = gates[Ops[idx][0]](i%(n_qubits//2))
                else:
                    O = gates["I"](i%(n_qubits//2))
                if i == n_qubits//2:
                    op = O
                else:
                    op = op@O
            op_list.append(op)
            if ((indices < N)*1.0).mean() == 1: # All indices in A
                local_operators_A.append(op_list)
            elif ((indices < N)*1.0).mean() == 0.: # All indices in B
                local_operators_B.append(op_list)
            else:
                overlap_operators.append(op_list)


H_A = 0j
for h in local_operators_A:
    H_A += h[0]*np.array(h[1].matrix())

H_B = 0j
for h in local_operators_B:
    H_B += h[0]*np.array(h[2].matrix())

H_A = qml.Hermitian(H_A, wires = range(n_qubits//2))
H_B = qml.Hermitian(H_B, wires = range(n_qubits//2))
H_AB = overlap_operators


H_overlap_A = [H_AB[i][1] for i in range(len(H_AB))]
H_overlap_B = [H_AB[i][2] for i in range(len(H_AB))]
H_overlap_coef = [H_AB[i][0] for i in range(len(H_AB))]

H_overlap_coef_jnp = jnp.array(H_overlap_coef)



#########    ARNN with netket     #########

alpha = 2
NN_features = alpha*2*N #dimension of the hidden neurons
NN_layers = 5 #nbr of layers

hi = nk.hilbert.Spin(s=0.5, N=int(2*N)) 
sa = nk.sampler.ARDirectSampler(hi) 
#sa = myARDirectSampler(hi) #if we want to control the number of one
model = nk.models.ARNNDense(hilbert=hi, layers= NN_layers, features=NN_features, param_dtype = np.float32)

# Initialize it
key = random.PRNGKey(123)
key, subkey = random.split(key)
s = jnp.ones(shape = (8, N), dtype=jnp.int32)
_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)






#########    ARNN Loss    #########


@jax.jit
def kernel(x,y):
  sigma = 1.0
  return jnp.prod(jnp.exp(-1*(x-y)**2/(2*sigma)))
  
@jax.jit
def Loss_ARNN(NN_params, set_bitstring_syst, lambdas):

  pi = jnp.exp(model.apply(NN_params, 2*(A_new-0.5)))

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
#optARNN = sgd(learning_rate=lr) 
opt_stateARNN = optARNN.init(NN_params)

swap_A = []
swap_B = []
history_loss = []


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
  
