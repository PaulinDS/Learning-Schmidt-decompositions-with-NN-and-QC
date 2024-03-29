![Image](figure_meta_algo.png)

# Learning-Schmidt-decompositions-with-NN-and-QC
Entanglement forging with autoregressive neural network (ARNN). In the Schrödinger forging scheme, an ARNN is used to identify the relevant bitstrings in the Schmidt decomposition. 

## Structure

In [config.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/config.py), there are the definitions of the Hamiltonian and the ARNN.

In [forging_functions.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/forging_functions.py), there are functions used for the entanglement forging.

In [generative_algo_functions.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/generative_algo_functions.py), there are functions used for the generative algorithm.

In [generative_algo_non_perm_sym.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/generative_algo_non_perm_sym.py), there is an example of how to use the different functions to run the algorithm generating the set the bitstrings on a non permutation symetric system.

In [schrodinger_forging_VQE.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/schrodinger_forging_VQE.py), there is an example of how to use the different functions to do the Schrodinger VQE. This part can be significantly accelerated with the use of GPUs.


## Package Versions

`pennylane 0.29.1`
`jax 0.3.16`
`jax_lib 0.3.15`
`optax 0.1.3`
`netket 3.8`
`openfermion 1.5.1`
`jaxopt 0.7`
