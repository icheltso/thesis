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
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
import timeit
from PIL import Image
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.numpy.fft import fft2, ifft2
import numpy as np

from skimage.util import random_noise

from runner import Runner
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D_jax, getWaveletTransforms_2D_jax, legend_without_duplicate_labels
import arviz as az
import seaborn as sns


'Setup up forward and inverse operators'
#p=1024
#p=32
s = 4
M = 1
#cam = pywt.data.camera()/255
cam = jnp.array(Image.open(os.path.join(parent_dir, 'lena128.jpg')).convert('L'), dtype=jnp.float64)/255
plt.imshow(cam, cmap="gray")
n,m = cam.shape
py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "haar", level = 4)
h = GaussianFilter_2D_jax(s,n,m)
#Phi = lambda x: jnp.real(ifft2(fft2(x)*fft2(h))); 
#Phi_s = lambda x: jnp.real(ifft2(fft2(x)*jnp.conjugate(fft2(h))))
Phi = lambda x: x
Phi_s = lambda x: x

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



#b = Phi(cam)
sigma = .05;
#b = random_noise(b,mode='gaussian',var=sigma,clip=False)

blotch_probability = 0.05  # Adjust this value for more or fewer blotches
mask = np.random.choice([0, 1], size=cam.shape, p=[blotch_probability, 1 - blotch_probability])

# Apply the mask to the image (blotches will be set to 0 - black)
blotched_image = cam * mask
b = blotched_image

plt.imshow(b, cmap="gray")

plt.show()

key = random.key(0)
key,subkey = random.split(key)
xinit_tile =  As(random.normal(key,[n,m,M]))
vinit_tile =  As(random.normal(subkey,[n,m,M]))



tau = 0.01
gamma = tau*.01
lam = 0.2
beta = 500
sig_noise = jnp.sqrt(2/beta)

b_tile = jnp.tile(b[:, :, jnp.newaxis], (1, 1, M))


"Parameter for truncated uv_CIR"
R = 1000

loss_type = "square_loss"

setup_data = Setup(sig_noise, loss_type, A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                'EULA',
                'uv_FB'
                #'uv_FB_trunc'
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_uvfb_trunc',
                #'one_part_eula',
                #'one_part_bessel'
                ]


### Visualization Params
n_bins = 30
no_stds = 15
#burn_in= 10**6
burn_in = 30000
#niter  = 5 * 10**6
niter = 50000
"Subsample every subsamp image"
subsamp= 100
thinned_steps = niter // subsamp


ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {m*n}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_2D", f"lam = {lam}, d = {m*n}")
start_time = timeit.default_timer()
x_out = Runner(setup_data, method_names).runner(niter, tau, burn_in, subsamp)
end_time = timeit.default_timer()
print('Methods took ', end_time - start_time, 'seconds')
for a in range(len(method_names)):
    plt.imshow(py_Ws(jnp.mean(x_out[a,:,:,:],axis=0)), cmap = "gray")
    plt.title("Mean reconstructed image: " + method_names[a])
    plt.pause(1)
    plt.clf()
    
    
"Apply backward wavelet transform to each element in chain"
x_out_py_Ws = []
for a in range(len(method_names)):
    x_out_fin = []
    for i in range(thinned_steps):
       if i%1000 == 0:
            print(i)
       x_out_fin.append(py_Ws(jnp.array(x_out[a,i,:,0])))
        
    x_out_py_Ws.append(jnp.reshape(jnp.array(x_out_fin), (1, thinned_steps, n,m)))
    

    
#palette_num = plt.get_cmap('tab10')
#fig_cred, ax_cred = plt.subplots(1, 3, figsize=(18, 6))
for a in range(len(method_names)):
    lower_bound = jnp.squeeze(jnp.percentile(x_out_py_Ws[a], 5, axis=1))
    upper_bound = jnp.squeeze(jnp.percentile(x_out_py_Ws[a], 95, axis=1))
    cred_width_sq = (upper_bound-lower_bound)**2
    plt.figure(figsize=(10, 8))
    plt.imshow(cred_width_sq, cmap='viridis')
    plt.colorbar(label='Square Width of 90% Credibility Interval')
    plt.title('Heatmap of 90% Squared Credibility Interval Width - ' + method_names[a])
    plt.show()
    #ax_cred.plot((upper_bound-lower_bound)**2, color = palette_num(a), label=method_names[a])
    
    
max_lag = 50
chosen_lag = 50
autocorr_data = []
acf_matrix = np.zeros((n,m, max_lag))
for a in range(len(method_names)):
    #acf_samples = jnp.squeeze(jnp.transpose(x_out[a], (2, 0, 1)))
    acf_samples = np.array(x_out_py_Ws[a][0])
    for d1 in range(n):
        for d2 in range(m):
            acf_matrix[d1,d2,:] = az.autocorr(acf_samples[:, d1,d2])[:max_lag]
    autocorr_data.append(acf_matrix)
    mean_acf = jnp.mean(acf_matrix, axis=(0,1))
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(acf_matrix[:,:,chosen_lag-1], cmap='coolwarm', cbar_kws={'label': 'Autocorrelation'}, 
                xticklabels=100, yticklabels=5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Autocorrelation Heatmap (in spatial domain) at lag ' + str(chosen_lag) + ' for ' + method_names[a])
    plt.show()
    
    #Plot Bar-Chart of mean ACF over dimension, similar to Figure 4 in Marcelo's paper.
    plt.figure(figsize=(10, 6))
    plt.bar(range(max_lag), mean_acf, width=1.0, color='blue', alpha=0.7)

    # Labeling the plot
    plt.xlabel('Lag')
    plt.ylabel('Mean Autocorrelation')
    plt.title('Mean Autocorrelation (in spatial domain) across dimensions for ' + method_names[a])
    plt.xticks(range(0, max_lag, 5))  # Show x-ticks at every 5th lag
    plt.grid(True, linestyle='--', alpha=0.6)
#ax_cred.set_title('Credibility Intervals')
#legend_without_duplicate_labels(ax_cred)
#visual = Visual(setup_data, method_names, x_arr, mean_arr, std_arr, tm_avg, ttl_xtra, save_xtra)
#visual.time_avg(0,niter-1)

#colours = get_N_HexCol(len(method_names))
#palette_num = plt.get_cmap('tab10')
#fig_wav, ax_wav = plt.subplots()
#x_out_py_Ws.append(jnp.reshape(jnp.array(x_out_fin), (1, niter, p)))





    

