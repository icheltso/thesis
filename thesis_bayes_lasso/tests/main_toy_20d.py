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

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import random
from scipy.integrate import nquad, quad

from runner import Runner
from visual import Visual
from setup import Setup
import arviz as az
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import create_matrix_cond

"Package relevant parameters into a single .npz file, which will be accessed by all other modules."
def setup_data(A,n,lam,x0,y,M,gamma,beta,key):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "data.pkl")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    key,subkey = random.split(key)
    #initx =  random.normal(key,[n,M])
    #initv =  random.normal(subkey,[n,M])
    
    
    init_all = random.normal(key,[3*n,M])
    
    initx = init_all[:n,:]
    initv = init_all[n:2*n,:]
    initz = init_all[2*n:,:]
    
    y2 = jnp.tile( y[:,None],(1,M))
    
    Af = lambda x: A @ x
    As = lambda x: A.T @ x
    
    'sde sigma'
    sigma = jnp.sqrt(2/beta)
    
    'Parameter R - for constrained uv method'
    R = 1000
    
    #todump = {A, As, n, lam, np.array(y), np.array(y2), np.array(initx), np.array(initv), M, gamma}
    
    'For loss_type, choose between square_loss and log_loss'
    loss_type = "square_loss"
    #loss_type = "log_loss"
    
    setup_data = Setup(sigma, loss_type, Af, As, n, lam, y, y2, initx, initv, initz, M, gamma, R)
    
    #np.savez(file_path, A=A, As=As, n=n, lam=lam, x0=x0, y=y, y2=y2, initx=initx, initv=initv, M=M, gamma=gamma)
    #with open(file_path, 'wb') as file:
    #    dill.dump(todump, file)
        
    return setup_data

def running_mean(array):
    running_means = np.zeros(len(array))
    cumulative_sum = 0

    for i, value in enumerate(array):
        cumulative_sum += value
        running_means[i] = cumulative_sum / (i + 1)

    return running_means

def exclude_nans(arr):
    """
    Excludes rows along the last dimension (x_part) that contain NaNs and tracks if any were removed.
    
    Parameters:
        arr (numpy.ndarray): Input array of shape (no_algs, subsamp_no, x_dim, x_part).
    
    Returns:
        filtered_arr (list): Nested list of NumPy arrays where NaN-containing rows in x_part are removed.
        nan_indicator (numpy.ndarray): Array of shape (no_algs,) with 1 if NaNs were found, else 0.
    """
    no_algs, subsamp_no, x_dim, x_part = arr.shape

    # Create a mask for valid (non-NaN) rows along the x_part dimension
    mask = ~np.any(np.isnan(arr), axis=-1)  # Shape: (no_algs, subsamp_no, x_dim)

    # Apply the mask to filter each individual sub-array
    filtered_arr = [
        [
            [
                arr[i, j, k, mask[i, j, k]]
                for k in range(x_dim)
            ]
            for j in range(subsamp_no)
        ]
        for i in range(no_algs)
    ]

    # Create a nan indicator array (1 if any NaNs were found, otherwise 0)
    nan_indicator = np.any(~mask, axis=(1, 2))  # Check if any row was removed per algorithm
    nan_indicator = nan_indicator.astype(int)   # Convert boolean to integer (1 or 0)

    return filtered_arr, nan_indicator

def aggregate_mean_OLD(runs,phi):
    runs_shape = runs.shape
    agg_mean = []
    #runs = jnp.array(runs)
    for i in range(runs_shape[0]):
        agg_mean_tmp = 0
        for j in range(runs_shape[3]):
            print('Aggregate Mean of Phi for Particle ' + str(j))
            running_mean = 0
            for k in range(runs_shape[1]):
                running_mean += phi(runs[i,k,:,j])
                
            agg_mean_tmp += running_mean / runs_shape[1]
            
        agg_mean.append(agg_mean_tmp / runs_shape[3])
        
    return agg_mean

def aggregate_mean_np(runs, phi):
    # Apply `phi` to all particles and samples
    phi_values = np.apply_along_axis(phi, 2, runs)  # Apply `phi` across the last dimension (dim)
    print('Applied phi')
    # Compute the mean over the sample dimension (axis=1) and particle dimension (axis=3)
    agg_mean = phi_values.mean(axis=(1, 3))
    return agg_mean

def aggregate_mean(runs, phi):
    # Apply phi to the dim axis (axis 2), not particles (axis 3)
    # Here, we apply phi over axis 2 (dim) which is where each x is a vector
    phi_values = jax.numpy.apply_along_axis(lambda x: phi(x), axis=2, arr=runs)  # Map phi over axis 2 (dim)
    
    # Check the shape of phi_values after vmap
    print("Shape of phi_values after vmap:", phi_values.shape)
    
    # Now average over niter (axis 1) and particles (axis 3)
    agg_mean = phi_values.mean(axis=(1, 2))  # Mean over niter (axis=1) and particles (axis=3)
    
    return agg_mean



##############################
###TESTS
##############################


method_names_main = ['PROXL1',
                'uv_FB',
                'cart_BOB',
                'cart_PEM',
                ]

method_names_gibbs = ['one_part_gibbs',]


#niter = 10**5

#burn_in = 10**5
tau_hada = 0.01
#gamma = tau*.2


#######Test parameters - CHANGE THESE
#lam_c = [0.01, 0.1]
#var_d = [1, 5, 50]
#cond_var = [1, 10, 100]
#######For bugfixin
lam_c = [0.1]
var_d = [1, 5, 10, 20]
cond_var = [1]

############################
#lam_c = [0]
#var_d = [5]
#cond_var = [1]
############################
#JAX random key, keep as is.
key = random.key(0)
#######For plotting histograms
n_bins = 30
no_stds = 15
m=10
#burn_in=100

#M = 10**2# number of particles
M = 1
beta = 1

#print('Hello World')

current_directory = os.getcwd()
file_path = os.path.join(current_directory, "data.npz")

runs_list = []

#Generate random matrix for toy problem
# Dimensions
n = 20
m = 40

# Desired variance
variance = 1 / (16 * m)
scale = jnp.sqrt(variance)




# Generate the matrix
key, subkey = random.split(key)
Amat = random.normal(subkey, shape=(m, n)) * scale
#Amat = np.random.normal(loc=0, scale=scale, size=(m, n)) * scale
#Amat = jnp.array([[-0.82127389]])
#Amat = jnp.array([[1]])
x0 = jnp.zeros((n,))
#x0 = x0.at[n//4].set(10)
#x0 = x0.at[n//2].set(3)
x0 = x0.at[0].set(10.0)
x0 = x0.at[1].set(3.0)
#x0 = jnp.array([-2])
#x0 = jnp.array([3])
y = Amat@x0
#y=[3]

true_vals = [x0[0], x0[1]]

lam = (1/2)*jnp.max(jnp.abs(Amat.T@y))
#lam = 2.7
#lam = 0.6 * jnp.abs(Amat @ y)
#lam = 0

#parameters for prox
Lf = np.linalg.norm(Amat)**2
gamma = 1 / (5*Lf)
tau_prox = 1 / (Lf + 1/gamma)

#tau = [tau_prox,
#       tau_hada
#       ]



#true_mean = get_quasi_true_mean(phi)

Amat_np = np.array(Amat)
ynp = np.array(y)


tau_val = 0.01
tau_main = [tau_val] * len(method_names_main)
tau_gibbs = [tau_val]

#T_burn = 800
#T_samp = 1000
#burn_in = int(T_burn / tau_val)
#niter = int(T_samp / tau_val)
burn_in = 10**4
niter = 10**5
subsamp = 10
M = 1

setup = setup_data(jnp.array(Amat), n, lam, jnp.array(x0), jnp.array(y), M, gamma, beta, key)

# Main methods: normal run
runout_main, burn_time_main, sample_time_main = Runner(setup, method_names_main).runner(
    niter, tau_main, burn_in, subsamp
)

# Gibbs: 10x fewer iterations, no subsampling
gibbs_niter = 10**4
gibbs_burn_in = 10
gibbs_subsamp = 1

runout_gibbs, burn_time_gibbs, sample_time_gibbs = Runner(setup, method_names_gibbs).runner(
    gibbs_niter, tau_gibbs, gibbs_burn_in, gibbs_subsamp
)

runouts = [runout_main[0], runout_main[1], runout_main[2], runout_main[3], runout_gibbs[0]]
plot_titles = ["Prox-L1", "Hadamard-UV", "Cartesian BOB", "Cartesian PEM", "Gibbs"]
method_names = method_names_main + method_names_gibbs

# runout shape should be: (methods, saved_iters, dim, particles)
informative_dims = [0, 1]

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})

fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharey=False)
axs = axs.ravel()

plot_start = 50
all_rm = []

#plot_titles = ["Prox-L1", "Hadamard-UV", "Cartesian BOB", "Cartesian PEM", "Gibbs"]
method_dict = {k: v for k, v in zip(method_names, plot_titles)}

for k, method in enumerate(plot_titles):
    ax = axs[k]

    for dim in range(n):
        #series = np.asarray(runout[k, :, dim, 0])
        series = np.asarray(runouts[k][:, dim, 0])
        rm = np.cumsum(series) / np.arange(1, len(series) + 1)
        rm_plot = rm[plot_start:]
        all_rm.append(rm_plot)

        if dim in informative_dims:
            ax.plot(np.arange(plot_start, len(rm)), rm_plot,
                    linewidth=2.4, label=f"Informative dim. {dim+1}")
        else:
            ax.plot(np.arange(plot_start, len(rm)), rm_plot,
                    linewidth=0.8, alpha=0.35)

    #if k == 0:
    #    ax.axhline(true_vals[0], linestyle=':', linewidth=2,
    #               color='C0', label='True value (dim. 1)')
    #    ax.axhline(true_vals[1], linestyle=':', linewidth=2,
    #               color='C1', label='True value (dim. 2)')
    #    ax.axhline(0, linestyle=':', linewidth=1.5,
    #               color='gray', label='Zero')
    #else:
    #    ax.axhline(true_vals[0], linestyle=':', linewidth=2, color='C0')
    #    ax.axhline(true_vals[1], linestyle=':', linewidth=2, color='C1')
    #    ax.axhline(0, linestyle=':', linewidth=1.5, color='gray')

    ax.set_title(method)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Running mean")
    if k == 0:
        ax.legend(fontsize=10)

# sane y-limits shared across running-mean panels only
all_rm_flat = np.concatenate(all_rm)
ymin, ymax = np.quantile(all_rm_flat, [0.005, 0.998])
pad = 0.15 * (ymax - ymin)

for ax in axs[:len(method_names)]:
    ax.set_ylim(ymin - pad, ymax + pad)

# use bottom-left as blank spacing
#axs[4].axis("off")

# ESS in bottom-right
ax = axs[5]
ess_vals = np.zeros((len(method_names), n))



for k in range(len(method_names)):
    for dim in range(n):
        #print(runout.shape)
        #print(runout[k, :, dim, 0].shape)
        #ess_vals[k, dim] = float(az.ess(np.asarray(runout[k, :, dim, 0])))
        #chain = np.asarray(runout[k, :, dim, 0])[None, :]  # shape: (1, draws)
        chain = np.asarray(runouts[k][:, dim, 0])[None, :]
        ess_vals[k, dim] = float(az.ess(chain))

x = np.arange(n)
width = 0.8 / len(method_names)

for k, method in enumerate(plot_titles):
    ax.bar(x + (k - (len(method_names)-1)/2) * width, ess_vals[k], width, label=method)

ax.set_title("ESS by dimension")
ax.set_xlabel("Dimension")
ax.set_ylabel("ESS")
ax.set_xticks(np.arange(0, n, 2))
ax.set_xticklabels(np.arange(1, n+1, 2))
ax.legend()

save_xtra = os.path.join("SIMULATION","TOY","20D")
os.makedirs(save_xtra, exist_ok=True)

fig.tight_layout()
plt.savefig(os.path.join(save_xtra, "figure_7_2_20d.pdf"), dpi=300, bbox_inches="tight")
plt.show()