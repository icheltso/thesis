# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:39:51 2024

@author: ichel
"""

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft, ifft
from PIL import Image
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
from jax.numpy.fft import fft2, ifft2


#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D, GaussianFilter_2D_jax, getWaveletTransforms_2D, getWaveletTransforms_2D_jax


p=16
#p=32
s = 4
M = 10
    
cam = jnp.array(Image.open(os.path.join(parent_dir, 'lena128.jpg')).convert('L'), dtype=jnp.float32)/255
plt.imshow(cam, cmap="gray")
n,m = cam.shape
py_W_jx, py_Ws_jx = getWaveletTransforms_2D_jax(n,m,wavelet_type = "haar", level = 4)
h = GaussianFilter_2D_jax(s,n,m)
Phi_jx = lambda x: jnp.real(ifft2(fft2(x)*fft2(h))); 
Phi_s_jx = lambda x: jnp.real(ifft2(fft2(x)*jnp.conjugate(fft2(h))))

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
#A_old = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
#As_old = lambda x: py_W(Phi_s(x))

def A_jx(coeffs):
    "Loop over M sets of coefficients"
    print(len(coeffs.shape))
    if len(coeffs.shape) == 1:
        return Phi_jx(py_Ws_jx(coeffs))
    xs = []
    #print("A input shape is " + str(coeffs.shape))
    for i in range(coeffs.shape[1]):
        #cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        xs.append(Phi_jx(py_Ws_jx(coeffs[:,i])))
        
    #print("Output of A has shape " + str(jnp.transpose(jnp.array(xs),(1,2,0)).shape))
        
    return jnp.transpose(jnp.array(xs),(1,2,0))

def As_jx(x):
    "Loop over M particles"
    print(len(x.shape))
    if len(x.shape) == 2:
        return py_W_jx(Phi_s_jx(x))
    cfs = []
    #print("As input shape is " + str(x.shape))
    for i in range(x.shape[2]):
        #x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        cfs.append(py_W_jx(Phi_s_jx(x[:,:,i])))
        
    #print("Output of As has shape " + str(jnp.transpose(jnp.array(cfs),(1,0)).shape))
        
    return jnp.transpose(jnp.array(cfs),(1,0))

"""Define numpy operators"""
py_W, py_Ws = getWaveletTransforms_2D(n,m,wavelet_type = "db2", level = 4)
h = GaussianFilter_2D(s,n,m)
Phi = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x)*np.fft.fft2(h))); 
Phi_s = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x)*np.conjugate(np.fft.fft2(h))))

'Set up operators A and A^-1'
# Phi o W^{-1}
A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
As = lambda x: py_W(Phi_s(x))


key = random.key(0)
key, subkey = random.split(key)

y_example = np.random.rand(n,m)
print("Wavelet Comparison")
x_example = py_W_jx(y_example)
x2_example = py_W(y_example)
print(jnp.allclose(x_example,x2_example))
print("Inverse Check - numpy")
print(jnp.allclose(y_example, py_Ws(x2_example)))
print(np.linalg.norm(y_example - py_Ws(py_W(y_example)),2) / np.linalg.norm(y_example,2))
print("Inverse Check - jax")
print(jnp.allclose(y_example, py_Ws_jx(py_W_jx(y_example))))
print(np.linalg.norm(y_example - py_Ws_jx(py_W_jx(y_example)),2) / np.linalg.norm(y_example,2))
print("Inverse of numpy vs inverse of jax")
print(np.linalg.norm(py_Ws_jx(py_W_jx(y_example)) - py_Ws(py_W(y_example)),2) / np.linalg.norm(py_Ws(py_W(y_example)),2))
print("\n A test - one particle")
np_A_onepart = A(x2_example)
jx_A_onepart = A_jx(x_example)
print(np.linalg.norm(np_A_onepart - jx_A_onepart,2) / np.linalg.norm(A(x2_example),2))

print("\n A/As test -  One-particle test. Compare py_Ws(As(A(py_W(x)))) for each implementation.")
np_WsAAsW_onepart = py_Ws(As(A(x2_example)))
jx_WsAAsW_onepart = py_Ws_jx(As_jx(A_jx(x_example)))
print(jnp.allclose(np_WsAAsW_onepart,jx_WsAAsW_onepart))
print(np.linalg.norm(np_WsAAsW_onepart - jx_WsAAsW_onepart,2))

print("\n Multi-particle tst")
x_tiled_jax = []
x_tiled = []
for i in range(M):
    randvar = np.random.rand(n,m)
    x_tiled.append(py_W(randvar))
    x_tiled_jax.append(py_W_jx(randvar))
    
x_tiled = jnp.array(x_tiled).T
x_tiled_jax = jnp.array(x_tiled_jax).T

numpy_AAs = []
for j in range(M):
    numpy_AAs.append(As(A(x_tiled[:,j])))
    
#numpy_AAs = jnp.array(numpy_AAs).T
print('here')
jax_AAs = As_jx(A_jx(x_tiled_jax))

np_WsAAsW_multipart = []
jx_WsAAsW_multipart = []
normdiff = []

for i in range(M):
    np_WsAAsW_multipart.append(py_Ws(numpy_AAs[i]))
    jx_WsAAsW_multipart.append(py_Ws_jx(jax_AAs[:,i]))
    normdiff.append(np.linalg.norm(np_WsAAsW_multipart[i] - jx_WsAAsW_multipart[i],2) / np.linalg.norm(np_WsAAsW_multipart[i],2))
    
np_WsAAsW_multipart = jnp.array(np_WsAAsW_multipart).T
jx_WsAAsW_multipart = jnp.array(jx_WsAAsW_multipart).T

print(jnp.allclose(jx_WsAAsW_multipart,np_WsAAsW_multipart))
print(max(normdiff))