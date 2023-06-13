import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import netket as nk
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_l2_sphere





#functions to construct the matrix for the syst of equation, Equation (...) in the paper, for permutation symetric systems
@jax.jit
def diago_terms_perm_sym(params_A, params_B, bitstring):
  A = Circuits_ObservableA(params_A, bitstring)
  B = Circuits_ObservableB(params_B, bitstring)
  O = jnp.sum(H_overlap_coef_jnp*Circuits_Observable_listA(params_A, bitstring)*Circuits_Observable_listB(params_B, bitstring))
  return 2.*(A + B + O)

@jax.jit
def off_diago_terms_perm_sym(params_A, params_B, bitstring_i, bitstring_j):
  p = jnp.arange(0,4)
  res = (-1)**p*Circuits_Observable_phi_jitA(params_A, bitstring_i ,bitstring_j, p)
  res += (-1)**p*Circuits_Observable_phi_jitB(params_B, bitstring_i ,bitstring_j, p)
  res += (-1)**p*jnp.einsum('i,ij,ij->j', H_overlap_coef_jnp, Circuits_Observable_phi_list_jitA(params_A, bitstring_i ,bitstring_j, p),Circuits_Observable_phi_list_jitB(params_B, bitstring_i ,bitstring_j, p))
  return jnp.sum(res)
  
diago_terms_perm_sym_vmap = jax.jit(jax.vmap(diago_terms_perm_sym, in_axes=(None,None,0)))
off_diago_terms_perm_sym_vmap = jax.jit(jax.vmap(jax.vmap(off_diago_terms_perm_sym, in_axes=(None,None,0,None)), in_axes=(None,None,None,0)))

def Generative_loop_perm_sym(NN_params, params_A, params_B, A):
  """
  steps 1-3 of the proposed algorithm (algo 1 in the paper), 
  1) generate a set G of bitstrings from the ARNN
  2) construct and solve the system of equations to compute the Schmidt coefficient
  3) Create the new set A'
  
  This version is for permutation symetric systems
  Thus, the same set of bitstrings for subsystem A and B can be taken
  
  NN_params: parameters of the ARNN
  params_A: parameters of the parametrized quantum circuit acting on subsystem A
  params_B: parameters of the parametrized quantum circuit acting on subsystem B
  A: current set of bitstring
  """

  #generate a set of bitstring
  cut_off = 8 #cutoff in the Schmidt decomposition
  chain_length = 10 #number of bitstrings generated

  s, G = get_sample(NN_params, chain_length=chain_length)

  #construct the matrix of the syst of eqs
  bitstring_syst = jnp.concatenate((A,G), axis=0)
  set_bitstring_syst, iS = jnp.unique(bitstring_syst, return_index=False, return_inverse=True, return_counts=False, axis=0)
  
  #to ensure that functions always have arguments with the same shape, we remove the repetition of bitstrings after constructing the matrix, otherwise it jits again every time.
  d = diago_terms_perm_sym_vmap(params_A, params_B, bitstring_syst)

  Matrix_syst = jnp.diag(d)

  off_d = off_diago_terms_perm_sym_vmap(params_A, params_B, bitstring_syst, bitstring_syst)
  cache = (jnp.triu(jnp.ones((jnp.shape(bitstring_syst)[0],jnp.shape(bitstring_syst)[0])))-1)*(-1) #creat a cache, triangular matrix with just 1 on the under tri
  Matrix_syst += cache*off_d

  Matrix_syst = Matrix_syst[jnp.ix_(iS,iS)] 

  #solve the constrain probl
  f = lambda x: jnp.linalg.norm(Matrix_syst@x)

  radius = 1.0
  pg = ProjectedGradient(fun=f, projection=projection_l2_sphere, maxiter=5000)
  lamb_init = random.uniform(subkey, [jnp.shape(Matrix_syst)[0]])
  lamb_init = lamb_init/jnp.sqrt(jnp.sum(lamb_init**2))
  lambdas = pg.run(lamb_init, hyperparams_proj=radius).params

  index_to_keep = jnp.argsort(jnp.abs(lambdas))[-cut_off:] 
  A_new = set_bitstring_syst[index_to_keep]
  lambdas_new = lambdas[index_to_keep]

  return A_new, set_bitstring_syst, lambdas, lambdas_new
  

#functions to construct the matrix for the syst of equation, Equation (...) in the paper, for non permutation symetric systems
@jax.jit
def diago_terms_non_perm_sym(params_A, params_B, sets):
    bitstringA, bitstringB = sets
    oA = Circuits_ObservableA(params_A, bitstringA)
    oB = Circuits_ObservableB(params_B, bitstringB)
    oO = jnp.sum(Circuits_Observable_listA(params_A, bitstringA)*Circuits_Observable_listB(params_B, bitstringB))
    return 2.*(oA + oB + oO)

@jax.jit
def off_diago_terms_non_perm_sym(params_A, params_B, sets_i, sets_j):
  bitstringA_i, bitstringB_i = sets_i
  bitstringA_j, bitstringB_j = sets_j
  p = jnp.arange(0,4)
  res = (-1)**p*Circuits_Observable_phi_jitA(params_A, bitstringA_i ,bitstringA_j, p)
  res += (-1)**p*Circuits_Observable_phi_jitB(params_B, bitstringB_i ,bitstringB_j, p)
  res += (-1)**p*jnp.einsum('i,ij,ij->j', H_overlap_coef_jnp, Circuits_Observable_phi_list_jitA(params_A, bitstringA_i ,bitstringA_j, p),Circuits_Observable_phi_list_jitB(params_B, bitstringB_i ,bitstringB_j, p))
  return jnp.sum(res)

diago_terms_non_perm_sym_vmap = jax.jit(jax.vmap(diago_terms_non_perm_sym, in_axes=(None,None,0)))
off_diago_terms_non_perm_sym_vmap = jax.jit(jax.vmap(jax.vmap(off_diago_terms_non_perm_sym, in_axes=(None,None,0,None)), in_axes=(None,None,None,0)))

@jax.jit
def split_gene(G_set):
  G_set_A, G_set_B = jnp.split(G_set, 2, axis=1)
  return G_set_A, G_set_B

@jax.jit
def split_new(new_set_bitstring_syst):
  A_new, B_new = jnp.split(new_set_bitstring_syst, 2, axis=1)
  return A_new, B_new



def Generative_loop_non_perm_sym(NN_params, params_A, params_B, A, B, key):

  """
  steps 1-3 of the proposed algorithm (algo 1 in the paper), 
  1) generate a set G of bitstrings from the ARNN
  2) construct and solve the system of equations to compute the Schmidt coefficient
  3) Create the new set A', B'
  
  This version is for non-permutation symetric systems
  Thus, the a different set of bitstrings for subsystem A and B need to be taken
  
  NN_params: parameters of the ARNN
  params_A: parameters of the parametrized quantum circuit acting on subsystem A
  params_B: parameters of the parametrized quantum circuit acting on subsystem B
  A: current set of bitstring of subsystem A
  B: current set of bitstring of subsystem B
  """

cut_off = 8
chain_length = 30

#generate a sets of bitstring
s, G_set = get_sample(NN_params, chain_length=chain_length)

G_A, G_B = split_gene(G_set)

#construct the matrix of the syst of eqs
bitstring_syst_A = jnp.concatenate((A,G_A), axis=0) #need to put A before G_A, otherwise, after we could get size < cutoff!
bitstring_syst_B = jnp.concatenate((B,G_B), axis=0) #since jnp.unique outputs the indexes based on the first occurrence, there's no problem if A/B are before 

_, iA = jnp.unique(bitstring_syst_A, return_index=True, return_inverse=False, return_counts=False, axis=0)
_, iB = jnp.unique(bitstring_syst_B, return_index=True, return_inverse=False, return_counts=False, axis=0)
iAB = jnp.intersect1d(iA, iB) #we need to remove the repetition of bitstrings in each subsystems
set_bitstring_syst_A = bitstring_syst_A[iAB]
set_bitstring_syst_B = bitstring_syst_B[iAB]


sets = (bitstring_syst_A,bitstring_syst_B)

d = diago_terms_non_perm_sym_vmap(params_A, params_B, sets)
Matrix_syst = jnp.diag(d)
off_d = off_diago_terms_non_perm_sym_vmap(params_A, params_B, sets, sets)
cache = 1-jnp.triu(jnp.ones((jnp.shape(bitstring_syst_A)[0],jnp.shape(bitstring_syst_A)[0]))) #creat a cache, triangular matrix with $
Matrix_syst += cache*off_d

#to ensure that functions always have arguments with the same shape, we remove the repetition of bitstrings after constructing the matrix, otherwise it jits again every time.
Matrix_syst = Matrix_syst[jnp.ix_(iAB,iAB)]


#solve the constrain probl
f = lambda x: jnp.linalg.norm(Matrix_syst@x)

radius = 1.0
pg = ProjectedGradient(fun=f, projection=projection_l2_sphere, maxiter=5000)
lamb_init = random.uniform(subkey, [jnp.shape(Matrix_syst)[0]])
lamb_init = lamb_init/jnp.sqrt(jnp.sum(lamb_init**2))
lambdas = pg.run(lamb_init, hyperparams_proj=radius).params

#reconc set A and B for the training of the ARNN
set_bitstring_syst = jnp.concatenate((set_bitstring_syst_A,set_bitstring_syst_B), axis=1)
#set_bitstring_syst, i = jnp.unique(bitstring_syst, return_index=True, return_inverse=False, return_counts=False, axis=0, size=None, fill_value=None)

index_to_keep = jnp.argsort(jnp.abs(lambdas))[-cut_off:]
new_set_bitstring_syst = set_bitstring_syst[index_to_keep]
#A_new, B_new = jnp.split(new_set_bitstring_syst, 2, axis=1)
A_new = set_bitstring_syst_A[index_to_keep]
B_new = set_bitstring_syst_B[index_to_keep]

return A_new, B_new, new_set_bitstring_syst, lambdas[index_to_keep]




  
