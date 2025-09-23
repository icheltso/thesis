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

"Investigate effect of varying lambda/beta on reconstruction and credibility intervals"


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

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D_jax, getWaveletTransforms_2D_jax, legend_without_duplicate_labels
import arviz as az
import seaborn as sns
import pywt


'Setup up forward and inverse operators'
#p=1024
#p=32
s = 2
M = 1
cam = pywt.data.camera()/255
#cam = jnp.array(Image.open(os.path.join(parent_dir, 'lena128.jpg')).convert('L'), dtype=jnp.float64)/255
fig_orig, ax_orig = plt.subplots()
ax_orig.imshow(cam, cmap="gray")
n,m = cam.shape
#py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "haar", level = 4)
py_W, py_Ws = getWaveletTransforms_2D_jax(n,m,wavelet_type = "db4", level = 6)
h = GaussianFilter_2D_jax(s,n,m)
Phi = lambda x: jnp.real(ifft2(fft2(x)*fft2(h))); 
Phi_s = lambda x: jnp.real(ifft2(fft2(x)*jnp.conjugate(fft2(h))))
b = Phi(cam)
sigma = .05;
b = random_noise(b,mode='gaussian',var=sigma,clip=False)

########## BLOTCH INPAINTING TEST
#Phi = lambda x: x
#Phi_s = lambda x: x
#blotch_probability = 0.2  # Adjust this value for more or fewer blotches
#mask = np.random.choice([0, 1], size=cam.shape, p=[blotch_probability, 1 - blotch_probability])
# Apply the mask to the image (blotches will be set to 0 - black)
#blotched_image = cam * mask
#b = blotched_image
##########
#inpainting
mask = np.random.rand(n,m) #random mask
mask = jnp.real(ifft2(fft2(mask)*fft2(h)))>0.48 #patchy mask
Phi = lambda x: mask*x
Phi_s = lambda x: mask*x
#observation
b = Phi(cam)
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


fig_mod,ax_mod = plt.subplots()
ax_mod.imshow(b, cmap="gray")

key = random.key(0)
key,subkey = random.split(key)
xinit_tile =  As(random.normal(key,[n,m,M]))
vinit_tile =  As(random.normal(subkey,[n,m,M]))

y = A(As(random.normal(key,[n,m,M])))

#tau = 0.01
#gamma = tau*.01
lam = 0.2
beta = 500
sig_noise = jnp.sqrt(2/beta)

b_tile = jnp.tile(b[:, :, jnp.newaxis], (1, 1, M))


"Parameter for truncated uv_CIR"
R = 1000

loss_type = "square_loss"

#setup_data = Setup(sig_noise, loss_type, A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)

method_names = [
                'EULA',
                #'uv_FB'
                #'uv_FB_trunc'
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_uvfb_trunc',
                #'one_part_eula',
                #'one_part_bessel'
                ]



#burn_in= 10**6
#niter  = 5 * 10**6
"Set stepsize as described in page 18 of paper https://arxiv.org/pdf/2201.09096"
Lf = np.linalg.norm(h)*len(cam)
gamma = 1 / (5*Lf)
tau = 1 / (Lf + 1/gamma)
burn_in = 30000
niter = 50000
#burn_in = 100
#niter = 1000

"Subsample every subsamp image"
subsamp= 100
thinned_steps = niter // subsamp

lam_base = jnp.max(jnp.abs(As(y)))

"Credibility range"
cred_upper = 95
cred_lower = 5

"Max lag for ACF calculations and chosen lag at which to plot."
max_lag = 25
chosen_lag = 10

save_stuff = False


#betavals = [1,5,50,500]
#lamfac = [0.01, 0.1, 1]

#betavals = [5,50,500]
#lamfac = [0.01,0.1,1]

betavals = [500]
lamfac = [0.001]

#betavals = [5]
#lamfac = [0.01]

"Variable for storing all runs over various parameter settings"
run_labels = []
"Variables for storing max/min credibility interval and ACF values, pixelwise. Needed for uniform heatmap ranges."
max_percentile = 0
min_percentile = 10**-6
max_acf = 0
min_acf = 10**-6
"Dictionaries for storing credibility intervals for various lambda/beta values."
cred_dict = {}
acf_dict = {}
for beta in betavals:
    sig_noise = jnp.sqrt(2/beta)
    for il in lamfac:
        run_labels.append(rf"$\lambda$-factor = {il}, $d$ = {n*m}, beta = {beta}")
        lam = float(lam_base*il)
        setup_data = Setup(sig_noise, loss_type, A, As, n**2, lam, b, b_tile, xinit_tile, vinit_tile, M, gamma, R)
        print("Started for beta = " + str(beta) + ", lambda = " + str(lam))
        ttl_xtra = rf"$\lambda$ = {lam}, $d$ = {n*m}, beta = {beta}"
        #Create string for saving to folders
        #save_xtra = os.path.join("SIMULATION","WAVELET_2D", f"burnin = {burn_in}, niter = {niter}, subsamp = {subsamp}, delta = {tau}, lam = {lam:.3g}, d = {n*m}, beta = {beta}")
        save_xtra = os.path.join("SIMULATION","WAVELET_2D", f"lam{il}beta{beta}")
        os.makedirs(save_xtra, exist_ok=True)
        #Save the original and blurred images to every subfolder
        filename_orig = "cam_original.png"
        fig_orig_path = os.path.join(save_xtra,filename_orig)
        fig_orig.savefig(fig_orig_path)
        filename_blur = "cam_blurred.png"
        fig_blur_path = os.path.join(save_xtra,filename_blur)
        fig_mod.savefig(fig_blur_path)
        start_time = timeit.default_timer()
        x_out = Runner(setup_data, method_names).runner(niter, tau, burn_in, subsamp)
        end_time = timeit.default_timer()
        x_out = jax.device_put(x_out, device=jax.devices("cpu")[0]) #<----- ON CPU
        # Clear GPU memory
        gc.collect()
        print('Methods took ', end_time - start_time, 'seconds')
        if save_stuff == True:
            for a in range(len(method_names)):
                filename_data = f"experiment_data_{method_names[a]}.npz"
                save_data_path = os.path.join(save_xtra, filename_data)
                np.savez_compressed(save_data_path, array = np.array(x_out[a]))
            
        print("Show mean reconstruction")
        for a in range(len(method_names)):
            filename = f"mean_rec_{method_names[a]}.png"
            fig_path = os.path.join(save_xtra,filename)
            plt.imshow(py_Ws(jnp.mean(x_out[a,:,:,:],axis=0)), cmap = "gray")
            plt.savefig(fig_path, bbox_inches='tight')
            plt.pause(1)
            plt.clf()
            
            # Clear GPU memory
            gc.collect()
            
        print("Applying backwards transform for credibility intervals")    
        x_out_py_Ws = []
        for a in range(len(method_names)):
            x_out_fin = []
            #x_out_gpu_chunk = jax.device_put(x_out[a], device=jax.devices("gpu")[0]) #<----- ON GPU
            for i in range(thinned_steps):
               if i%100 == 0:
                    print(rf"Applied backward transform to {i}  of {thinned_steps} elements.")
               x_out_fin.append(py_Ws(jnp.array(x_out[a,i,:,0])))
               #x_out_fin.append(py_Ws(jnp.array(x_out_gpu_chunk[i,:,0])))    #<----- ON GPU
            # Clear GPU memory
            gc.collect()
            if save_stuff == True:
                "Also save backward transformed data"
                filename_data = f"spatial_domain_{method_names[a]}.npz"
                save_data_path = os.path.join(save_xtra, filename_data)
                np.savez_compressed(save_data_path, array = np.array(x_out_fin))
            x_out_py_Ws.append(jnp.reshape(jnp.array(x_out_fin), (1, thinned_steps, n,m)))      #<----- SHOULD BE ON CPU
            
        del x_out, x_out_fin
        gc.collect()
            
        print("Calculating credibility intervals and acf values")
        for a in range(len(method_names)):
            lower_bound = np.squeeze(np.percentile(x_out_py_Ws[a], cred_lower, axis=1))
            upper_bound = np.squeeze(np.percentile(x_out_py_Ws[a], cred_upper, axis=1))
            cred_width_sq = (upper_bound-lower_bound)**2
            max_percentile = max(max_percentile,np.max(cred_width_sq))
            min_percentile = min(min_percentile,np.min(cred_width_sq))
            print("Save cred ints to dictionary")
            cred_dict[(beta,il,a)] = cred_width_sq
            
            print("Calculating acf values")
            acf_samples = np.array(x_out_py_Ws[a][0])
            acf_matrix = np.zeros((n,m, max_lag))
            print("Obtain ACF matrix at lag " + str(max_lag))
            for d1 in range(n):
                for d2 in range(m):
                    acf_matrix[d1,d2,:] = az.autocorr(acf_samples[:, d1,d2])[:max_lag]
            print("Save acf data to dictionary")
            acf_dict[(beta,il,a)] = acf_matrix
            max_acf = max(max_acf,jnp.max(acf_matrix))
            min_acf = min(min_acf,jnp.min(acf_matrix))
            
        del x_out_py_Ws
        gc.collect()
            
            
            
"Plot Relevant Heatmaps"
for beta in betavals:
    for il in lamfac:
        for a in range(len(method_names)):
            "Credibility interval plot"
            fig_cred,ax_cred = plt.subplots()
            sns.heatmap(np.log(cred_dict[beta,il,a]),ax=ax_cred, vmin = np.log(min_percentile), vmax = np.log(max_percentile),  cmap='plasma')
            filename_cred = f"cred_{cred_upper - cred_lower}_{method_names[a]}.png"
            fig_path = os.path.join(save_xtra,filename_cred)
            fig_cred.savefig(fig_path)
            plt.show()
            
            "ACF plot"
            fig_acf,ax_acf = plt.subplots()
            sns.heatmap(acf_dict[beta,il,a][:,:,chosen_lag-1], ax=ax_acf, vmin = min_acf, vmax = max_acf,  cmap='plasma')
            filename_acf = f"acf_lag_{chosen_lag}_{method_names[a]}.png"
            fig_path = os.path.join(save_xtra,filename_acf)
            fig_acf.savefig(fig_path)
            plt.show()
            
            
"Mean ACF Plot"
#Plot Bar-Chart of mean ACF over dimension, similar to Figure 4 in Marcelo's paper.
lags = np.arange(max_lag)
mean_acf = []
for a in range(len(method_names)):
    mean_acf.append(np.mean(acf_dict[beta,il,a], axis=(0,1)))
colors = plt.cm.get_cmap('viridis', len(method_names))  # Color map for `num_datasets` colors
fig_acfm, ax_acfm = plt.subplots()

#plt.bar(range(max_lag), mean_acf, width=1.0, color='blue', alpha=0.7)
bar_width = 0.15


# Plot each dataset with an offset
for i, (data, color) in enumerate(zip(mean_acf, colors.colors)):
    # Offset the x positions for each dataset
    ax_acfm.bar(lags + i * bar_width - (len(method_names) - 1) * bar_width / 2,
           data,
           width=bar_width,
           label=method_names[i],
           color=color,
           alpha=0.7)  # Set transparency

# Adding labels and title
ax_acfm.set_xlabel('Lag')
ax_acfm.set_ylabel('ACF')
# Adding a legend
ax_acfm.legend()
# Show plot


filename_acfm = f"acf_mean_{max_lag}.png"
fig_path_acfm = os.path.join(save_xtra,filename_acfm)
fig_acfm.savefig(fig_path_acfm)
plt.show()




    

