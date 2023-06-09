import jax
import jax.numpy as jnp
from jax import random
import pennylane as qml
from functools import partial
import netket as nk
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_l2_sphere


def sample_NN(NN_params, chain_length = 20, sa = None, NN_model = None, n_qubits = 10):
    """
    get sample from NN in $s \in {-1, 1}$ and $S \in {0,1}$ conversion.
    NN_params: Parameters of classical model
    chain_length: Number of samples
    sa: Netket sampler
    NN_model: Netket NN model
    """
    Sample, _ = nk.sampler.ARDirectSampler.sample(sa, NN_model, NN_params, chain_length = chain_length)
    #Sample, _ = myARDirectSampler.sample(sa, NN_model, NN_params, chain_length = chain_length)
    Sample = Sample.reshape(-1, n_qubits//2)
    s = jax.lax.stop_gradient(Sample)
    S = (s + 1)/2
    S = S.astype(int)
    return s, S

get_sample = partial(sample_NN, sa = sa, NN_model = model, n_qubits = n_qubits)


#fct to construct the matrix for the syst of eqs
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
  
diago_terms_perm_sym_vmap = jax.jit(jax.vmap(diago_perm_sym_terms, in_axes=(None,None,0)))
off_diago_terms_perm_sym_vmap = jax.jit(jax.vmap(jax.vmap(off_diago_terms, in_axes=(None,None,0,None)), in_axes=(None,None,None,0)))

def Generative_loop_perm_sym(NN_params, params_A, params_B, A):

  #generate a set of bitstring
  cut_off = 8
  chain_length = 10

  s, G = get_sample(NN_params, chain_length=chain_length)

  #construct the matrix of the syst of eqs
  bitstring_syst = jnp.concatenate((A,G), axis=0)
  set_bitstring_syst, iS = jnp.unique(bitstring_syst, return_index=False, return_inverse=True, return_counts=False, axis=0)
  
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
  #lambdas_G = lambdas[index_G]

  return A_new, set_bitstring_syst, lambdas, lambdas_new
  


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
    This function is for non symmetric systems
    """

    cut_off = 8
    chain_length = 30

    #generate a sets of bitstring
    s, G_set = get_sample(NN_params, chain_length=chain_length)

    G_A, G_B = split_gene(G_set)

    #construct the matrix of the syst of eqs
    bitstring_syst_A = jnp.concatenate((A,G_A), axis=0) #important de mettre A avant G_A, sinon apres on peut avoir des sets de taille < cutoff!
    bitstring_syst_B = jnp.concatenate((B,G_B), axis=0) #etant donné que jnp.unique sort les indexes de la premiere occurence, pas de soucis si c'est avant

    _, iA = jnp.unique(bitstring_syst_A, return_index=True, return_inverse=False, return_counts=False, axis=0)
    _, iB = jnp.unique(bitstring_syst_B, return_index=True, return_inverse=False, return_counts=False, axis=0)
    iAB = jnp.intersect1d(iA, iB)
    set_bitstring_syst_A = bitstring_syst_A[iAB]
    set_bitstring_syst_B = bitstring_syst_B[iAB]

    #index_G = i[:jnp.shape(G_set_A)[0]]
    #sets = (set_bitstring_syst_A,set_bitstring_syst_B)
    sets = (bitstring_syst_A,bitstring_syst_B)

    d = diago_terms_non_perm_sym_vmap(params_A, params_B, sets)
    Matrix_syst = jnp.diag(d)
    off_d = off_diago_terms_non_perm_sym_vmap(params_A, params_B, sets, sets)
    cache = 1-jnp.triu(jnp.ones((jnp.shape(bitstring_syst_A)[0],jnp.shape(bitstring_syst_A)[0]))) #creat a cache, triangular matrix with $
    Matrix_syst += cache*off_d

    #remove the repetition of the bitstrings (after the creation of M to jit)
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

    return A_new, B_new, set_bitstring_syst, lambdas, new_set_bitstring_syst, lambdas[index_to_keep]




  
