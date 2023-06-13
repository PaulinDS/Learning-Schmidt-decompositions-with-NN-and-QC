import jax
import jax.numpy as jnp
import pennylane as qml
from jax.lax import scan
from jax.lax import cond
import numpy as np

import config


def brick_wall_entangling(params):
    """
    One layer of the parametrized quantum circuit
    """
    layers, qubits, _ = params.shape
    for i in range(qubits):
        qml.Hadamard(wires = i)
    for i in range(layers):
        for j in range(qubits):
            qml.Rot(*params[i][j], wires = j)
        for j in range(int(qubits/2)):
            if i%2 == 0:
                qml.CNOT(wires = [2*j, 2*j+1])
            if i%2 == 1:
                qml.CNOT(wires = [2*j + 1, (2*j+2)%qubits])

                
 ##########      Function evaluating the various quantum expectation values in equation (...) in the paper  ##########               


# In order not to have the operators in argument and being able to jit the functions, a lot of function have been define
# Thus, the operators have to be global
# Need functions for local operators in A, local operators in B and overlapping operators 

@jax.jit
def Circuits_ObservableA(params, inputs):
    dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
    @jax.jit   
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return qml.expval(config.H_A) 
    return qnode(params, inputs)

@jax.jit
def Circuits_ObservableB(params, inputs):
    dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return qml.expval(config.H_B)
    return qnode(params, inputs) 

@jax.jit
def Circuits_Observable_listA(params, inputs):
    dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
    @jax.jit  
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(config.n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return [qml.expval(Obs) for Obs in config.H_overlap_B]
    return qnode(params, inputs) 

@jax.jit
def Circuits_Observable_listB(params, inputs):
    dev = qml.device('default.qubit.jax', wires=config.n_qubits//2) 
    @jax.jit   
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(config.n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return [qml.expval(Obs) for Obs in config.H_overlap_B]
    return qnode(params, inputs) 
   

#functions preparing the state |phi> following the steps of S.M. 3 in https://arxiv.org/abs/2104.10220
    
@jax.jit
def Circuits_Observable_phi_jitA(params, inputs_n ,inputs_m, p):  

  k = jnp.nonzero(inputs_n != inputs_m, size=1)[0][0]

  new_p = (-1)**inputs_n[k]*p%4
  new_inputs_n = inputs_n[k]*inputs_m + (1-inputs_n[k])*inputs_n
  new_inputs_m = inputs_n[k]*inputs_n + (1-inputs_n[k])*inputs_m

  S = jnp.nonzero(new_inputs_n != new_inputs_m, size=N)[0][1:]
  T = jnp.nonzero(new_inputs_n == 1, size=N)[0]
  t0 = new_inputs_n[0]

  dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
  @jax.jit
  @qml.qnode(dev, interface='jax', diff_method="backprop")
  def qnode(params, T, S, t0):

      #X
      qml.RX(phi=((t0==1)*jnp.pi), wires=0)
      for t in T:
        for i in range(1,N):
          qml.RX(phi=((t==i)*jnp.pi), wires=i)
      
      for i in range(N):
        qml.Rot(phi=((k==i)*jnp.pi),theta=((k==i)*jnp.pi/2),omega=(jnp.pi*0),wires=i) #H
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([2,3]))*(k==i)*jnp.pi, wires=i) #Z
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([1,3]))*(k==i)*jnp.pi/2, wires=i) #S

      #CNOT
      for l in S:
        for i in range(N):
          for j in range(i+1,N):
              qml.CRZ(jnp.pi*(k==i)*(l==j), wires=[i,j])
              qml.CRot(jnp.pi/2*(k==i)*(l==j), jnp.pi*(k==i)*(l==j), -jnp.pi/2*(k==i)*(l==j), wires=[i,j])
              qml.ControlledPhaseShift(jnp.pi*(k==i)*(l==j), wires=[i,j])

      brick_wall_entangling(params)
      
      return qml.expval(config.H_A)  

  return qnode(params, T, S, t0)

@jax.jit
def Circuits_Observable_phi_jitB(params, inputs_n, inputs_m, p):  

  k = jnp.nonzero(inputs_n != inputs_m, size=1)[0][0]

  new_p = (-1)**inputs_n[k]*p%4
  new_inputs_n = inputs_n[k]*inputs_m + (1-inputs_n[k])*inputs_n
  new_inputs_m = inputs_n[k]*inputs_n + (1-inputs_n[k])*inputs_m

  S = jnp.nonzero(new_inputs_n != new_inputs_m, size=N)[0][1:]
  T = jnp.nonzero(new_inputs_n == 1, size=N)[0]
  t0 = new_inputs_n[0]

  dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
  @jax.jit 
  @qml.qnode(dev, interface='jax', diff_method="backprop")
  def qnode(params, T, S, t0):

      #X
      qml.RX(phi=((t0==1)*jnp.pi), wires=0)
      for t in T:
        for i in range(1,N):
          qml.RX(phi=((t==i)*jnp.pi), wires=i)
      
      for i in range(N):
        qml.Rot(phi=((k==i)*jnp.pi),theta=((k==i)*jnp.pi/2),omega=(jnp.pi*0),wires=i) #H
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([2,3]))*(k==i)*jnp.pi, wires=i) #Z
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([1,3]))*(k==i)*jnp.pi/2, wires=i) #S

      #CNOT
      for l in S:
        for i in range(N):
          for j in range(i+1,N):
              qml.CRZ(jnp.pi*(k==i)*(l==j), wires=[i,j])
              qml.CRot(jnp.pi/2*(k==i)*(l==j), jnp.pi*(k==i)*(l==j), -jnp.pi/2*(k==i)*(l==j), wires=[i,j])
              qml.ControlledPhaseShift(jnp.pi*(k==i)*(l==j), wires=[i,j])

      brick_wall_entangling(params)
      
      return qml.expval(config.H_B)  #qml.state()

  return qnode(params, T, S, t0)

@jax.jit
def Circuits_Observable_phi_list_jitA(params, inputs_n, inputs_m, p):
  k = jnp.nonzero(inputs_n != inputs_m, size=1)[0][0]

  new_p = (-1)**inputs_n[k]*p%4
  new_inputs_n = inputs_n[k]*inputs_m + (1-inputs_n[k])*inputs_n
  new_inputs_m = inputs_n[k]*inputs_n + (1-inputs_n[k])*inputs_m

  S = jnp.nonzero(new_inputs_n != new_inputs_m, size=N)[0][1:]
  T = jnp.nonzero(new_inputs_n == 1, size=N)[0]
  t0 = new_inputs_n[0]

  dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
  @jax.jit
  @qml.qnode(dev, interface='jax', diff_method="backprop")
  def qnode(params, T, S, t0):

      #X
      qml.RX(phi=((t0==1)*jnp.pi), wires=0)
      for t in T:
        for i in range(1,N):
          qml.RX(phi=((t==i)*jnp.pi), wires=i)
      
      for i in range(N):
        qml.Rot(phi=((k==i)*jnp.pi),theta=((k==i)*jnp.pi/2),omega=(jnp.pi*0),wires=i) #H
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([2,3]))*(k==i)*jnp.pi, wires=i) #Z
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([1,3]))*(k==i)*jnp.pi/2, wires=i) #S

      #CNOT
      for l in S:
        for i in range(N):
          for j in range(i+1,N):
              qml.CRZ(jnp.pi*(k==i)*(l==j), wires=[i,j])
              qml.CRot(jnp.pi/2*(k==i)*(l==j), jnp.pi*(k==i)*(l==j), -jnp.pi/2*(k==i)*(l==j), wires=[i,j])
              qml.ControlledPhaseShift(jnp.pi*(k==i)*(l==j), wires=[i,j]) 

      brick_wall_entangling(params)

      return [qml.expval(Obs) for Obs in config.H_overlap_A]

  return qnode(params, T, S, t0)

@jax.jit
def Circuits_Observable_phi_list_jitB(params, inputs_n, inputs_m, p):
  k = jnp.nonzero(inputs_n != inputs_m, size=1)[0][0]

  new_p = (-1)**inputs_n[k]*p%4
  new_inputs_n = inputs_n[k]*inputs_m + (1-inputs_n[k])*inputs_n
  new_inputs_m = inputs_n[k]*inputs_n + (1-inputs_n[k])*inputs_m

  S = jnp.nonzero(new_inputs_n != new_inputs_m, size=N)[0][1:]
  T = jnp.nonzero(new_inputs_n == 1, size=N)[0]
  t0 = new_inputs_n[0]

  dev = qml.device('default.qubit.jax', wires=config.n_qubits//2)  
  @jax.jit 
  @qml.qnode(dev, interface='jax', diff_method="backprop")
  def qnode(params, T, S, t0):

      #X
      qml.RX(phi=((t0==1)*jnp.pi), wires=0)
      for t in T:
        for i in range(1,N):
          qml.RX(phi=((t==i)*jnp.pi), wires=i)
      
      for i in range(N):
        qml.Rot(phi=((k==i)*jnp.pi),theta=((k==i)*jnp.pi/2),omega=0,wires=i) #H
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([2,3]))*(k==i)*jnp.pi, wires=i) #Z
      for i in range(N):
        qml.PhaseShift(jnp.isin(new_p,jnp.array([1,3]))*(k==i)*jnp.pi/2, wires=i) #S

      #CNOT
      for l in S:
        for i in range(N):
          for j in range(i+1,N):
              qml.CRZ(jnp.pi*(k==i)*(l==j), wires=[i,j])
              qml.CRot(jnp.pi/2*(k==i)*(l==j), jnp.pi*(k==i)*(l==j), -jnp.pi/2*(k==i)*(l==j), wires=[i,j])
              qml.ControlledPhaseShift(jnp.pi*(k==i)*(l==j), wires=[i,j]) 

      brick_wall_entangling(params)
      
      return [qml.expval(Obs) for Obs in config.H_overlap_B]

  return qnode(params, T, S, t0)



def energy_vmap2(params_A, params_B, Schmidt_coef, bitstringA, bitstringB):
  """
  Compute the variational energy using the functions define above
  
  params_A: parameters of the variational circuit acting on subsystem A
  params_B: parameters of the variational circuit acting on subsystem B
  Schmidt_coef: Schmidt coefficients, jnp vector if size (cutoff)
  bitstringA: bitstrings of subsystem A, jnp matrix of size (cutoff,N)
  bitstringB: bitstrings of subsystem B, jnp matrix of size (cutoff,N)
  """

  E = 0
  S_rank = jnp.shape(bitstringA)[0] 
  
  #all diago terms
  Circ_Obs_partA_vmap = jax.vmap(partial(Circuits_ObservableA, params=params_A)) 
  Circ_Obs_partB_vmap = jax.vmap(partial(Circuits_ObservableB, params=params_B))

  Circ_Obs_part_overA_vmap = jax.vmap(partial(Circuits_Observable_listA, params=params_A))
  Circ_Obs_part_overB_vmap = jax.vmap(partial(Circuits_Observable_listB, params=params_B))

  E += jnp.sum(Schmidt_coef*Schmidt_coef*Circ_Obs_partA_vmap(inputs=bitstringA))
  E += jnp.sum(Schmidt_coef*Schmidt_coef*Circ_Obs_partB_vmap(inputs=bitstringB))

  A = Circ_Obs_part_overA_vmap(inputs=bitstringA)
  B = Circ_Obs_part_overB_vmap(inputs=bitstringB)
  E += jnp.sum(Schmidt_coef*Schmidt_coef*jnp.sum(H_overlap_coef_jnp*A*B,axis=1))


  ##off diago terms
  #loc op
  p = jnp.arange(0,4,1,dtype=int)

  Circ_Obs_phi_partA_vmap = jax.vmap(jax.vmap(partial(Circuits_Observable_phi_jitA, params_A),in_axes=(0, None, None)), in_axes=(None, 0, None))
  resA = Circ_Obs_phi_partA_vmap(bitstringA, bitstringA, p)
  Circ_Obs_phi_partB_vmap = jax.vmap(jax.vmap(partial(Circuits_Observable_phi_jitB, params_B),in_axes=(0, None, None)), in_axes=(None, 0, None))
  resB = Circ_Obs_phi_partB_vmap(bitstringB, bitstringB, p)
  Circ_Obs_phi_part_overA_vmap = jax.vmap(jax.vmap(partial(Circuits_Observable_phi_list_jitA, params_A),in_axes=(0, None, None)), in_axes=(None, 0, None))
  resoA = Circ_Obs_phi_part_overA_vmap(bitstringA, bitstringA, p)
  Circ_Obs_phi_part_overB_vmap = jax.vmap(jax.vmap(partial(Circuits_Observable_phi_list_jitB, params_B),in_axes=(0, None, None)), in_axes=(None, 0, None))
  resoB = Circ_Obs_phi_part_overB_vmap(bitstringB, bitstringB, p)

 # using jax.lax.scan but I think it could also be possible to do just tensors constractions (cleaner)
  def Eloc_loop_offdiag(carry,m): 
    Et, n = carry

    Et += Schmidt_coef[n]*Schmidt_coef[m]*jnp.sum((-1)**p*resA[n,m])*cond(m<n, lambda x: 1, lambda x: 0, 0) #to have the sum m<n
    Et += Schmidt_coef[n]*Schmidt_coef[m]*jnp.sum((-1)**p*resB[n,m])*cond(m<n, lambda x: 1, lambda x: 0, 0)
    Et += Schmidt_coef[n]*Schmidt_coef[m]*jnp.sum((-1)**p*jnp.einsum('k,ijkl,ijkl->ijl',H_overlap_coef_jnp,resoA,resoB)[n,m])*cond(m<n, lambda x: 1, lambda x: 0, 0)

    return (Et, n), m

  def Eloc_loop(carry,n): 
    Et = carry
    temp, m = scan(Eloc_loop_offdiag, init=(0,n), xs=jnp.arange(0,S_rank,1,dtype=int))
    Ett, n = temp
    Et += Ett
    return Et, n

  Et, n = scan(Eloc_loop, init=0, xs=jnp.arange(0,S_rank,1,dtype=int))
  E += Et

  
  return E 


grad_E_fn_circA = jax.jit(jax.value_and_grad(energy_vmap2, argnums = 0))
grad_E_fn_circB = jax.jit(jax.value_and_grad(energy_vmap2, argnums = 1))
grad_E_fn_schmidt = jax.jit(jax.value_and_grad(energy_vmap2, argnums = 2))



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



