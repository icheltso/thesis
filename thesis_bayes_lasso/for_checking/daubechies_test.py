# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:15:45 2024

@author: ichel
"""

import numpy as np
import pywt
from util import getWaveletTransforms_2D_jax, GaussianFilter_2D_jax

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
import time
import timeit
from PIL import Image
import gc
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.numpy.fft import fft2, ifft2
import numpy as np
########DEBUG
#from jax.config import config
#jax.config.update("jax_platform_name", "gpu")  # Ensure GPU is being used
#jax.config.update("jax_debug_nans", True)     # Detect NaNs
#jax.config.update("jax_xla_backend", "gpu")  # XLA GPU backend

#import os
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable preallocation
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'   # Use only 80% of VRAM
############

from skimage.util import random_noise

from runner import Runner
from setup import Setup

n = 512
m = 512
s = 4

# Example: Create a 2D random signal (image)
image = np.random.randn(n,m)  # 2D signal of size 128x128
print("Original Image shape:", image.shape)

# Perform multiple levels of decomposition
levels = 6
coeffs_haar_multi = pywt.wavedec2(image, 'haar', level=levels)
coeffs_db4_multi = pywt.wavedec2(image, 'db4', level=levels)

py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "db4", level = 6)

# Print shapes of coefficients for Haar (db1)
print("\nHaar Multi-level Coefficients:")
for i, coeff in enumerate(coeffs_haar_multi):
    if i == 0:
        print(f"Level {i} Approximation shape: {coeff.shape}")
    else:
        print(f"Level {i} Horizontal detail shape: {coeff[0].shape}")
        print(f"Level {i} Vertical detail shape: {coeff[1].shape}")
        print(f"Level {i} Diagonal detail shape: {coeff[2].shape}")

# Print shapes of coefficients for Daubechies-4 (db4)
print("\nDaubechies-4 (db4) Multi-level Coefficients:")
for i, coeff in enumerate(coeffs_db4_multi):
    if i == 0:
        print(f"Level {i} Approximation shape: {coeff.shape}")
    else:
        print(f"Level {i} Horizontal detail shape: {coeff[0].shape}")
        print(f"Level {i} Vertical detail shape: {coeff[1].shape}")
        print(f"Level {i} Diagonal detail shape: {coeff[2].shape}")
        
print(py_Ws(py_W(image)) - image)

s=4
h = GaussianFilter_2D_jax(s,n,m)

mask = np.random.rand(n,m) #random mask
mask = jnp.real(ifft2(fft2(mask)*fft2(h)))>0.48 #patchy mask
Phi = lambda x: mask*x
Phi_s = lambda x: mask*x
#observation
b = Phi(image)
sigma = .01;
b = random_noise(b,mode='gaussian',var=sigma,clip=False) # add noise

'Set up operators A and A^-1'
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
y = A(As(random.normal(key,[n,m,1])))
print(y.shape)
