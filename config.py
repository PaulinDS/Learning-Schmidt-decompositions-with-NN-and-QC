import openfermion as of
from openfermion import QubitOperator
import jax
import jax.numpy as jnp
from jax import random
import pennylane as qml
import numpy as np
import netket as nk

#########    Define the Hamiltonian     #########

n_qubits = 12
key = random.PRNGKey(1234)
key, subkey = random.split(key)
h_i = random.uniform(subkey, (n_qubits,), dtype = jnp.float32).tolist()
H_of = QubitOperator('Z{} Z0'.format(n_qubits-1))
for i in range(n_qubits):
  H_of += QubitOperator('Z{} Z{}'.format(i, i+1)) 
  H_of += h_i[0]*QubitOperator('X{}'.format(i))
  

N = n_qubits//2

perm_sym = True #if the system is invariant under the permutation of the 2 subsystems



#########    Define the ARNN with netket     #########


alpha = 2
NN_layers = 5 #nbr of layers

if perm_sym == True:
  NN_features = alpha*N #dimension of the hidden neurons
  hi = nk.hilbert.Spin(s=0.5, N=int(N)) 
else:
  NN_features = alpha*2*N #dimension of the hidden neurons
  hi = nk.hilbert.Spin(s=0.5, N=int(2*N)) 

sa = nk.sampler.ARDirectSampler(hi) 
#sa = myARDirectSampler(hi) #if we want to control the number of one
model = nk.models.ARNNDense(hilbert=hi, layers= NN_layers, features=NN_features, param_dtype = np.float32)




#########    Convert the Hamiltonian     #########

def creat_EF_hamiltonian(H_of, n_qubits):

  """
  group the operators into overlapping and local ones, convert them into qml, code taken and adapted from:
  https://github.com/cqsl/Entanglement-Forging-with-GNN-models

  H_of: Hamiltonian of the system expressed with openfermion
  n_qubits: number of qubits of the total system
  """

  gates = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "I":qml.Identity}

  op_list_list = []

  local_operators_A = []
  local_operators_B = []
  overlap_operators = []

  N = n_qubits//2

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

  return H_A, H_B, H_overlap_A, H_overlap_B, H_overlap_coef_jnp

H_A, H_B, H_overlap_A, H_overlap_B, H_overlap_coef_jnp = creat_EF_hamiltonian(H_of, n_qubits)

