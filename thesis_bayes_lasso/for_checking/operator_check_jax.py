# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:17:51 2024

@author: ichel
"""
import os
os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft, ifft
import time
from PIL import Image
import pywt
import jax.numpy as jnp
import jax
from jax import random
from jax import jit
from jax.numpy.fft import fft, ifft

from skimage.util import random_noise

from runner import Runner
from visual import Visual
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import create_matrix_cond, GaussianFilter, getWaveletTransforms_jax, get_N_HexCol, legend_without_duplicate_labels
#
import matplotlib.pyplot as plt


p=1024
#p=32
s = 4
M = 10
h = GaussianFilter(s,p)
Phi = lambda x: jnp.real(ifft(fft(x)*fft(h))); 
Phi_s = lambda x: jnp.real(ifft(fft(x)*jnp.conjugate(fft(h))))
    
#mask = np.zeros(p)
#mask[:p//8]=np.ones(p//8)
#mask[p-p//8:]+=1
#mask[np.random.permutation(p)[:p//4]] =  np.ones(p//4)

#Phi = lambda x:np.concatenate( ( np.real(mask*fft(x)), np.imag(mask*fft(x))))/np.sqrt(p)
#Phi_s = lambda x: np.real(ifft(mask*(np.array(np.array(x)[:p] + 1j*np.array(x)[p:])))*np.sqrt(p))

lev = int(np.log2(p))-1
py_W, py_Ws = getWaveletTransforms_jax(p,wavelet_type = "haar",level = lev)

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
#A_old = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
#As_old = lambda x: py_W(Phi_s(x))

def A(coeffs):
    "Loop over M sets of coefficients"
    if len(coeffs.shape) == 1:
        return Phi(py_Ws(coeffs))
    cf_out = jnp.copy(coeffs)
    for i in range(coeffs.shape[1]):
        cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        
    return cf_out

def As(x):
    "Loop over M particles"
    if len(x.shape) == 1:
        return py_W(Phi_s(x))
    x_out = jnp.copy(x)
    for i in range(x.shape[1]):
        x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        
    return x_out


key = random.key(0)
key, subkey = random.split(key)

y_example = jax.random.uniform(key, (p,))
x_example = py_W(jax.random.uniform(subkey, (p,)))
print("\n Jax wavelets: ")

x_Tadj_y = jnp.dot(x_example, jnp.conjugate(py_W(y_example)))
T_x_y = jnp.dot(py_Ws(x_example).flatten(), jnp.conjugate(y_example.flatten()))
print(jnp.allclose(x_Tadj_y, T_x_y))
print("\n Inverse from image to image:")
print(jnp.allclose(py_Ws(py_W(y_example)), y_example))
print("\n ||py_Ws(py_W(y)) - y||_2 = " + str(jnp.linalg.norm(py_Ws(py_W(y_example)) - y_example,2)) )
print("\n Inverse from coefficients to coefficients:")
print(jnp.allclose(py_W(py_Ws(x_example)), x_example))
print("\n ||py_W(py_Ws(x)) - x||_2 = " + str(jnp.linalg.norm(py_W(py_Ws(x_example)) - x_example,2)) )

print("\n A, As operators (jax) ")

print("\n Adjoint:")
x_Tadj_y = jnp.dot(x_example, jnp.conjugate(As(y_example)))
T_x_y = jnp.dot(A(x_example).flatten(), jnp.conjugate(y_example.flatten()))
print(jnp.allclose(x_Tadj_y, T_x_y))
print("\n Inverse from image to image:")
print(jnp.allclose(A(As(y_example)), y_example))
print("\n ||py_Ws(py_W(y)) - y||_2 = " + str(jnp.linalg.norm(A(As(y_example)) - y_example,2)) )
print("\n Inverse from coefficients to coefficients:")
print(jnp.allclose(As(A(x_example)), x_example))
print("\n ||py_W(py_Ws(x)) - x||_2 = " + str(jnp.linalg.norm(As(A(x_example)) - x_example,2)) )