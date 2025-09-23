# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:34 2024

@author: ichel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""

import os
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
import timeit
from PIL import Image

from skimage.util import random_noise

from runner import Runner
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D, getWaveletTransforms_2D



'Setup up forward and inverse operators'
p=1024
#p=32
s = 4
M = 1
#cam = pywt.data.camera()/255
cam = np.array(Image.open('lena128.jpg').convert('L'))/255
n,m = cam.shape
py_W, py_Ws = getWaveletTransforms_2D(n,m,wavelet_type = "db2", level = 4)
h = GaussianFilter_2D(s,n,m)
Phi = lambda x: np.real(ifft2(fft2(x)*fft2(h))); 
Phi_s = lambda x: np.real(ifft2(fft2(x)*np.conjugate(fft2(h))))

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
As = lambda x: py_W(Phi_s(x))


b = Phi(cam)
sigma = .05;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)

plt.imshow(b, cmap="gray")

plt.show()



R = 100

#tau = 1/20
tau = 0.01
gamma = tau*.01
lam = 0.2
sig_noise = np.sqrt(2)


xinit_tile = []
vinit_tile = []
for i in range(M):
    xinit_tile.append(As(np.random.normal(size = (n,m))))
    vinit_tile.append(As(np.random.normal(size = (n,m))))
    
xinit_tile = np.array(xinit_tile).T
vinit_tile = np.array(vinit_tile).T

b_tile = np.tile(b,(M,1)).T

setup_data = Setup(sig_noise, A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                #'one_part_mala',
                'one_part_uvfb',
                #'one_part_eula',
                #'one_part_bessel'
                ]


### Visualization Params
n_bins = 30
no_stds = 15
m=40
burn_in=100000

niter = 500

ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {p}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_1D", f"lam = {lam}, d = {p}")

start_time = timeit.default_timer()
x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
end_time = timeit.default_timer()
print('Methods took ', end_time - start_time, 'seconds')

for a in range(len(method_names)):
   plt.imshow(py_Ws(tm_avg[a,-1]), cmap = "gray")
   plt.pause(1)
   plt.clf()
plt.show()






    

