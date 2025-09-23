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
import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft, ifft
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import arviz as az
from jax.numpy.fft import fft, ifft
from skimage.util import random_noise
from runner import Runner
from setup import Setup
from util import GaussianFilter, getWaveletTransforms_jax, legend_without_duplicate_labels
import seaborn as sns

'Setup up forward and inverse operators'
p=1024
#p=32
s = 4
M = 1
h = GaussianFilter(s,p)
Phi = lambda x: jnp.real(ifft(fft(x)*fft(h))); 
Phi_s = lambda x: jnp.real(ifft(fft(x)*jnp.conjugate(fft(h))))
    

lev = int(np.log2(p))-1
py_W, py_Ws = getWaveletTransforms_jax(p,wavelet_type = "haar",level = lev)

def A(coeffs):
    "Loop over M sets of coefficients"
    if len(coeffs.shape) == 1:
        return Phi(py_Ws(coeffs))
    xs = []
    for i in range(coeffs.shape[1]):
        #cf_out.at[:,i].set(Phi(py_Ws(coeffs[:,i])))
        xs.append(Phi(py_Ws(coeffs[:,i])))
        
    return jnp.array(xs).T

def As(x):
    "Loop over M particles"
    if len(x.shape) == 1:
        return py_W(Phi_s(x))
    cfs = []
    for i in range(x.shape[1]):
        #x_out.at[:,i].set(py_W(Phi_s(x_out[:,i])))
        cfs.append(py_W(Phi_s(x[:,i])))
        
    return jnp.array(cfs).T

    

t = jnp.linspace(-2.5, 2.5, p)
x0 = jnp.piecewise(t, [t < -1.5, t >= 0,t>1], [-2, 4,-2])
b = Phi(x0)
sigma = .001;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)



tau = 0.01
gamma = tau*.01
lam = 1
#print("Hello World")
xinit = As(b)
vinit = jnp.ones_like(xinit)

key = random.key(0)
key,subkey = random.split(key)
xinit_tile =  As(random.normal(key,[p,M]))
vinit_tile =  As(random.normal(subkey,[p,M]))

#xinit_tile = jnp.tile(xinit,(M,1)).T
#vinit_tile = jnp.tile(vinit,(M,1)).T
b_tile = jnp.tile(b,(M,1)).T

R = 10000
sig_noise = jnp.sqrt(2)
#sig_noise = jnp.sqrt(0.1)

#loss_type = "log_loss"
loss_type = "square_loss"

setup_data = Setup(sig_noise, loss_type, A, As, p, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                #'one_part_mala',
                #'one_part_uvfb',
                #'uv_FB_trunc',
                'uv_FB',
                'EULA',
                #'MASCIR'
                #'one_part_eula',
                #'one_part_bessel'
                ]


### Visualization Params
n_bins = 30
no_stds = 15
m=40
burn_in=5000
niter = 10**5
subsamp = 10
subsamp_niter = niter // subsamp

ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {p}"
#Create string for saving to folders
save_xtra = os.path.join("WAVELET_1D", f"lam = {lam}, d = {p}")

#x_arr, mean_arr, std_arr, tm_avg = Runner(setup_data, niter, tau, method_names).runner(burn_in)
x_out = Runner(setup_data, method_names).runner(niter, tau, burn_in, subsamp)
#visual = Visual(setup_data, method_names, x_arr, mean_arr, std_arr, tm_avg, ttl_xtra, save_xtra)
#visual.time_avg(0,niter-1)


palette_num = plt.get_cmap('tab10')
fig_wav, ax_wav = plt.subplots()

for a in range(len(method_names)):
    #ax_wav.plot(mean_arr[a,-1], color = palette_num(a), label=method_names[a])
    ax_wav.plot(py_Ws(x_out[a,-1,:,0]), color = palette_num(a), label=method_names[a])


ax_wav.plot(Phi_s(b), color = palette_num(len(method_names)), label='Phi^-1(y)')
ax_wav.plot(x0, 'k', label='target')
ax_wav.set_title('x0 vs samples')
legend_without_duplicate_labels(ax_wav)


"Display mean/variance, compared to ground truth"
fig_wav_moments, ax_wav_moments = plt.subplots()

for a in range(len(method_names)):
    #ax_wav.plot(mean_arr[a,-1], color = palette_num(a), label=method_names[a])
    ax_wav_moments.plot(py_Ws(np.mean(x_out[a,:,:,0], axis=0)), color = palette_num(a), label=method_names[a])
    #ax_wav_moments.plot(py_Ws(np.var(x_out[a,:,:,0], axis=0)), linestyle=':', color = palette_num(a), label=method_names[a])
    "Haven't figured out how to plot variance data correctly"

#ax_wav_moments.plot(Phi_s(b), color = palette_num(len(method_names)), label='Phi^-1(y)')
ax_wav_moments.plot(x0, 'k', label='target')
ax_wav_moments.set_title('x0 vs sample means')
legend_without_duplicate_labels(ax_wav_moments)


"Apply backward wavelet transform to each element in chain"
x_out_py_Ws = []
for a in range(len(method_names)):
    x_out_fin = []
    for i in range(subsamp_niter):
       if i%5000 == 0:
            print(i)
       x_out_fin.append(py_Ws(x_out[a,i,:,0]))
        
    x_out_py_Ws.append(np.reshape(np.array(x_out_fin), (1, subsamp_niter, p)))
    
# Calculate 90% credibility intervals (pixel-wise)
fig_cred, ax_cred = plt.subplots()
for a in range(len(method_names)):
    lower_bound = np.squeeze(np.percentile(x_out_py_Ws[a], 5, axis=1))
    upper_bound = np.squeeze(np.percentile(x_out_py_Ws[a], 95, axis=1))
    ax_cred.plot((upper_bound-lower_bound)**2, color = palette_num(a), label=method_names[a])
    
ax_cred.set_title('Credibility Intervals')
legend_without_duplicate_labels(ax_cred)

fig_az, ax_az = plt.subplots()
#samples_for_arviz = []
max_lag = 50
autocorr_data = []
autocorr_spat_data = []
acf_matrix = np.zeros((p, max_lag))
acf_spat_matrix = np.zeros((p, max_lag))
for a in range(len(method_names)):
    acf_samples = np.squeeze(np.transpose(x_out[a], (2, 0, 1)))
    acf_spat_samples = np.array(x_out_py_Ws[a][0])
    for d in range(p):
        acf_matrix[d] = az.autocorr(acf_samples[:, d])[:max_lag]
        acf_spat_matrix[d] = az.autocorr(acf_samples[:, d])[:max_lag]
    autocorr_data.append(acf_matrix)
    autocorr_spat_data.append(acf_spat_matrix)
    mean_acf = np.mean(acf_matrix, axis=0)
    mean_spat_acf = np.mean(acf_spat_matrix, axis=0)
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(acf_matrix.T, cmap='coolwarm', cbar_kws={'label': 'Autocorrelation'}, 
                xticklabels=100, yticklabels=5)
    plt.xlabel('Dimension Index')
    plt.ylabel('Lag')
    plt.title('Autocorrelation heatmap across wavelet domain for ' + method_names[a])
    plt.show()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(acf_spat_matrix.T, cmap='coolwarm', cbar_kws={'label': 'Autocorrelation'}, 
                xticklabels=100, yticklabels=5)
    plt.xlabel('Dimension Index')
    plt.ylabel('Lag')
    plt.title('Autocorrelation heatmap across spatial domain for ' + method_names[a])
    plt.show()
    
    #Plot Bar-Chart of mean ACF over dimension, similar to Figure 4 in Marcelo's paper.
    plt.figure(figsize=(10, 6))
    plt.bar(range(max_lag), mean_acf, width=1.0, color='blue', alpha=0.7)

    # Labeling the plot
    plt.xlabel('Lag')
    plt.ylabel('Mean Autocorrelation')
    plt.title('Mean autocorrelation across wavelet domain for ' + method_names[a])
    plt.xticks(range(0, max_lag, 5))  # Show x-ticks at every 5th lag
    plt.grid(True, linestyle='--', alpha=0.6)
    
    #Plot Bar-Chart of mean ACF over spatial dimension, similar to Figure 4 in Marcelo's paper.
    plt.figure(figsize=(10, 6))
    plt.bar(range(max_lag), mean_spat_acf, width=1.0, color='blue', alpha=0.7)

    # Labeling the plot
    plt.xlabel('Lag')
    plt.ylabel('Mean Autocorrelation')
    plt.title('Mean autocorrelation across spatial domain for ' + method_names[a])
    plt.xticks(range(0, max_lag, 5))  # Show x-ticks at every 5th lag
    plt.grid(True, linestyle='--', alpha=0.6)
    
    


#for a in range(len(method_names)):
#    samples_for_arviz.append(np.transpose(x_out[a], (2, 0, 1)))
#    idata = az.convert_to_dataset(np.transpose(x_out[a], (2, 0, 1)))
#    acf_values = az.autocorr(idata)
#    ax_az.plot(acf_values, color = palette_num(a), label=method_names[a])
    
#ax_az.set_title('Autocorrelation')
#legend_without_duplicate_labels(ax_az)

#ess_vals = []
#fig_bar, ax_bar = plt.subplots()

#for a in range(len(method_names)):
#    samples_for_ess.append(np.transpose(x_out[a], (2, 0, 1)))
#    idata = az.convert_to_inference_data(samples_for_ess)
    #ax_wav.bar(method_names, , color = palette_num(a), label=method_names[a])
    #ess_vals.append(az.ess(x_out[a,:,:,0].T))


#ax_bar.bar(method_names, ess_vals, color = palette_num(len(method_names)))
#ax_bar.set_title('ESS')




    

