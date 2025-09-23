# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:17:51 2024

@author: ichel
"""
import os
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
jax.config.update("jax_enable_x64", True)
from jax import random
from jax import jit
from jax.numpy.fft import fft2, ifft2

from skimage.util import random_noise

from runner import Runner
from visual import Visual
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import create_matrix_cond, GaussianFilter_2D_jax, getWaveletTransforms_2D_jax, get_N_HexCol, legend_without_duplicate_labels
#
import matplotlib.pyplot as plt


s = 4
M = 1
#cam = pywt.data.camera()/255
cam = jnp.array(Image.open(os.path.join(parent_dir, 'lena128.jpg')).convert('L'), dtype=jnp.float32)
plt.imshow(cam, cmap="gray")
n,m = cam.shape
py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "haar", level = 4)
h = GaussianFilter_2D_jax(s,n,m)
Phi = lambda x: jnp.real(ifft2(fft2(x)*fft2(h))); 
Phi_s = lambda x: jnp.real(ifft2(fft2(x)*jnp.conjugate(fft2(h))))

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
#A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
#As = lambda x: py_W(Phi_s(x))


def A(coeffs):
    "Loop over M sets of coefficients"
    #print(len(coeffs.shape))
    if len(coeffs.shape) == 1:
        return Phi(py_Ws(coeffs))
    xs = []
    #print("A input shape is " + str(coeffs.shape))
    for i in range(coeffs.shape[1]):
        #cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        xs.append(Phi(py_Ws(coeffs[:,i])))
        
    #print("Output of A has shape " + str(jnp.transpose(jnp.array(xs),(1,2,0)).shape))
        
    return jnp.transpose(jnp.array(xs),(1,2,0))

def As(x):
    "Loop over M particles"
    #print(len(x.shape))
    if len(x.shape) == 2:
        return py_W(Phi_s(x))
    cfs = []
    #print("As input shape is " + str(x.shape))
    for i in range(x.shape[2]):
        #x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        cfs.append(py_W(Phi_s(x[:,:,i])))
        
    #print("Output of As has shape " + str(jnp.transpose(jnp.array(cfs),(1,0)).shape))
        
    return jnp.transpose(jnp.array(cfs),(1,0))


key = random.key(0)
key, subkey = random.split(key)

y_example = jax.random.uniform(key, shape=(n,m))
x_example = py_W(jax.random.uniform(subkey, shape=(n,m)))
print("\n Jax wavelets:")
print("\n Adjoint:")
x_Tadj_y = jnp.dot(x_example, jnp.conjugate(py_W(y_example)))
T_x_y = jnp.dot(py_Ws(x_example).flatten(), jnp.conjugate(y_example.flatten()))
print(jnp.allclose(x_Tadj_y, T_x_y))
print("\n Inverse from image to image:")
print(jnp.allclose(py_Ws(py_W(y_example)), y_example))
print("\n ||py_Ws(py_W(y)) - y||_2 = " + str(jnp.linalg.norm(py_Ws(py_W(y_example)) - y_example,2)) )
print("\n Inverse from coefficients to coefficients:")
print(jnp.allclose(py_W(py_Ws(x_example)), x_example))
print("\n ||py_W(py_Ws(x)) - x||_2 = " + str(jnp.linalg.norm(py_W(py_Ws(x_example)) - x_example,2)) )