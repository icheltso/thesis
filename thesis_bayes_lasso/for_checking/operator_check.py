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
from jax import random
from jax import jit
from numpy.fft import fft, ifft

from skimage.util import random_noise

from runner import Runner
from visual import Visual
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import create_matrix_cond, GaussianFilter, getWaveletTransforms, get_N_HexCol, legend_without_duplicate_labels
#
import matplotlib.pyplot as plt


p=1024
#p=32
s = 4
M = 1
h = GaussianFilter(s,p)
Phi = lambda x: np.real(ifft(fft(x)*fft(h))); 
Phi_s = lambda x: np.real(ifft(fft(x)*np.conjugate(fft(h))))
    
#mask = np.zeros(p)
#mask[:p//8]=np.ones(p//8)
#mask[p-p//8:]+=1
#mask[np.random.permutation(p)[:p//4]] =  np.ones(p//4)

#Phi = lambda x:np.concatenate( ( np.real(mask*fft(x)), np.imag(mask*fft(x))))/np.sqrt(p)
#Phi_s = lambda x: np.real(ifft(mask*(np.array(np.array(x)[:p] + 1j*np.array(x)[p:])))*np.sqrt(p))

lev = int(np.log2(p))-1
py_W, py_Ws = getWaveletTransforms(p,wavelet_type = "haar",level = lev)

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
As = lambda x: py_W(Phi_s(x))


y_example = np.random.rand(p)
x_example = py_W(np.random.rand(p))
print("\n Numpy wavelets:")
print("\n Adjoint:")
x_Tadj_y = np.dot(x_example, np.conjugate(py_W(y_example)))
T_x_y = np.dot(py_Ws(x_example).flatten(), np.conjugate(y_example.flatten()))
print(np.allclose(x_Tadj_y, T_x_y))
print("\n Inverse from image to image:")
print(np.allclose(py_Ws(py_W(y_example)), y_example))
print("\n ||py_Ws(py_W(y)) - y||_2 = " + str(np.linalg.norm(py_Ws(py_W(y_example)) - y_example,2)) )
print("\n Inverse from coefficients to coefficients:")
print(np.allclose(py_W(py_Ws(x_example)), x_example))
print("\n ||py_W(py_Ws(x)) - x||_2 = " + str(np.linalg.norm(py_W(py_Ws(x_example)) - x_example,2)) )

print("\n A, As operators (jax) ")

print("\n Adjoint:")
x_Tadj_y = np.dot(x_example, np.conjugate(As(y_example)))
T_x_y = np.dot(A(x_example).flatten(), np.conjugate(y_example.flatten()))
print(np.allclose(x_Tadj_y, T_x_y))
print("\n Inverse from image to image:")
print(np.allclose(A(As(y_example)), y_example))
print("\n ||py_Ws(py_W(y)) - y||_2 = " + str(np.linalg.norm(A(As(y_example)) - y_example,2)) )
print("\n Inverse from coefficients to coefficients:")
print(np.allclose(As(A(x_example)), x_example))
print("\n ||py_W(py_Ws(x)) - x||_2 = " + str(np.linalg.norm(As(A(x_example)) - x_example,2)) )