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
from numpy.fft import fft, ifft

from skimage.util import random_noise

from runner import Runner
from visual import Visual
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter, getWaveletTransforms, legend_without_duplicate_labels

'Setup up forward and inverse operators'
p=1024
#p=32
s = 4
M = 1
h = GaussianFilter(s,p)
#Phi = lambda x: np.real(ifft(fft(x)*fft(h))); 
#Phi_s = lambda x: np.real(ifft(fft(x)*np.conjugate(fft(h))))
    
mask = np.zeros(p)
mask[:p//8]=np.ones(p//8)
mask[p-p//8:]+=1
mask[np.random.permutation(p)[:p//4]] =  np.ones(p//4)

Phi = lambda x:np.concatenate( ( np.real(mask*fft(x)), np.imag(mask*fft(x))))/np.sqrt(p)
Phi_s = lambda x: np.real(ifft(mask*(np.array(np.array(x)[:p] + 1j*np.array(x)[p:])))*np.sqrt(p))

lev = int(np.log2(p))-1
py_W, py_Ws = getWaveletTransforms(p,wavelet_type = "haar",level = lev)

'Set up operators A and A^-1'
# Phi o W^{-1}
#A = lambda coeffs: Phi(py_Ws(coeffs)).reshape(-1,1)
A = lambda coeffs: Phi(py_Ws(coeffs))
# W o Phi 
#As = lambda x: py_W(Phi_s(x.squeeze())).reshape(-1,1)
As = lambda x: py_W(Phi_s(x))

t = np.linspace(-2.5, 2.5, p)
x0 = np.piecewise(t, [t < -1.5, t >= 0,t>1], [-2, 4,-2])
b = Phi(x0)
sigma = .001;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)


R = 100
tau = 0.001
gamma = tau*.01
lam = 1
sig_noise = np.sqrt(2)
#print("Hello World")
xinit = As(b)
vinit = np.ones_like(xinit)

xinit_tile = np.tile(xinit,(M,1)).T
vinit_tile = np.tile(vinit,(M,1)).T
b_tile = np.tile(b,(M,1)).T

setup_data = Setup(sig_noise, A, As, p, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

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

niter = 1000

ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {p}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_1D", f"lam = {lam}, d = {p}")


x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
visual = Visual(setup_data, method_names, x_arr, mean_arr, std_arr, tm_avg, ttl_xtra, save_xtra)
visual.time_avg(0,niter-1)

#colours = get_N_HexCol(len(method_names))
palette_num = plt.get_cmap('tab10')
fig_wav, ax_wav = plt.subplots()

for a in range(len(method_names)):
    ax_wav.plot(py_Ws(tm_avg[a,-1]), color = palette_num(a), label=method_names[a])


ax_wav.plot(Phi_s(b), color = palette_num(len(method_names)), label='Phi^-1(y)')
ax_wav.plot(x0, 'k', label='target')
ax_wav.set_title('x0 vs samples')
legend_without_duplicate_labels(ax_wav)




    

