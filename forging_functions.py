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
from functools import partial
from jax.lax import scan
from jax.lax import cond
from jax.lax import dynamic_slice
import optax
from optax import adabelief, noisy_sgd, yogi, adam, sgd
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_l2_sphere

#SU
def brick_wall_entangling(params):
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


@jax.jit
def Circuits_ObservableA(params, inputs):
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
    @jax.jit   
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return qml.expval(H_A)
    return qnode(params, inputs)

@jax.jit
def Circuits_ObservableB(params, inputs):
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return qml.expval(H_B)
    return qnode(params, inputs) 

@jax.jit
def Circuits_Observable_listA(params, inputs):
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
    @jax.jit  
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return [qml.expval(Obs) for Obs in H_overlap_B]
    return qnode(params, inputs) 

@jax.jit
def Circuits_Observable_listB(params, inputs):
    dev = qml.device('default.qubit.jax', wires=n_qubits//2) 
    @jax.jit   
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        for i in range(n_qubits//2):
          qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return [qml.expval(Obs) for Obs in H_overlap_B]
    return qnode(params, inputs) 
   
@jax.jit
def Circuits_Observable_phi_jitA(params, inputs_n ,inputs_m, p):  

  k = jnp.nonzero(inputs_n != inputs_m, size=1)[0][0]

  new_p = (-1)**inputs_n[k]*p%4
  new_inputs_n = inputs_n[k]*inputs_m + (1-inputs_n[k])*inputs_n
  new_inputs_m = inputs_n[k]*inputs_n + (1-inputs_n[k])*inputs_m

  S = jnp.nonzero(new_inputs_n != new_inputs_m, size=N)[0][1:]
  T = jnp.nonzero(new_inputs_n == 1, size=N)[0]
  t0 = new_inputs_n[0]

  dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
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
      
      return qml.expval(H_A)  #qml.state()

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

  dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
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
      
      return qml.expval(H_B)  #qml.state()

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

  dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
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

      return [qml.expval(Obs) for Obs in H_overlap_A]

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

  dev = qml.device('default.qubit.jax', wires=n_qubits//2)  
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
      
      return [qml.expval(Obs) for Obs in H_overlap_B]

  return qnode(params, T, S, t0)



