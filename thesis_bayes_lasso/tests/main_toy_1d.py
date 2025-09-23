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


method_names = ['PROXL1',
                #'EULA',
                'uv_FB',
                'cart_BOB',
                #'cart_seq',
                #'cart_EM',
                #'cart_OBO',
                'cart_PEM'
                #'EULA_FB',
                #'uv_Bessel',
                #'MASCIR',
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_eula',
                #'one_part_bessel'
                #'one_part_gibbs',
                ]

#method_names = [#'EULA',
                #'uv_FB',
                #'cart_BOB',
                #'cart_seq',
                #'cart_EM',
                #'cart_OBO',
                #'cart_PEM'
                #'EULA_FB',
                #'uv_Bessel',
                #'MASCIR',
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_eula',
                #'one_part_bessel'
                #'one_part_gibbs',
                #]

no_methods = len(method_names)
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
M = 20
beta = 1

#print('Hello World')

current_directory = os.getcwd()
file_path = os.path.join(current_directory, "data.npz")

runs_list = []

#Generate random matrix for toy problem
# Dimensions
n = 1
m = 1

# Desired variance
variance = 1 / (16 * m)
scale = jnp.sqrt(variance)

# Generate the matrix
#Amat = np.random.normal(loc=0, scale=scale, size=(m, n))
#Amat = jnp.array([[-0.82127389]])
Amat = jnp.array([[1]])
#x0 = jnp.zeros((n,))
#x0 = x0.at[n//4].set(10)
#x0 = x0.at[n//2].set(3)
#x0 = jnp.array([-2])
x0 = jnp.array([3])
y = Amat@x0
#y=[3]

#lam = (1/2)*jnp.max(jnp.abs(Amat.T@y))
lam = 2.7
#lam = 0.6 * jnp.abs(Amat @ y)
#lam = 0

#parameters for prox
Lf = np.linalg.norm(Amat)**2
gamma = 1 / (5*Lf)
tau_prox = 1 / (Lf + 1/gamma)

#tau = [tau_prox,
#       tau_hada
#       ]

#phi = lambda x: np.linalg.norm(x,2)**2
phi = lambda x: jnp.sum(x**2)
phi_np = lambda x: np.sum(x**2)

"Run many realizations of hadamard at small stepsize to approximate true dist."
def get_quasi_true_mean(phi_func):
    print("Obtain quasi-mean using many iterations of Hadamard")
    method_base = ["uv_FB"]
    taus = [5*10**(-4)]
    burn_in_base = 0
    M_base = 5*10**4
    T = 2000
    niter_base = int(T/taus[0])
    #niter_base = 10**4
    subsamp_base = niter_base
    setup_base = setup_data(jnp.array(Amat),n,lam,jnp.array(x0),jnp.array(y),M_base,gamma,beta,random.key(1))
    runout_base = Runner(setup_base, method_base).runner(niter_base, taus, burn_in_base, subsamp_base) 
    runout_base = runout_base[0,-1,:,:]
    phi_run = []
    for i in range(M_base):
        phi_run.append(phi_func(runout_base[:,i]))
        
    phi_run = np.array(phi_run)
    mean_run = np.mean(phi_run)
    return mean_run

#true_mean = get_quasi_true_mean(phi)

Amat_np = np.array(Amat)
ynp = np.array(y)

"Calculate integral numerically, using integrate.nquad. For high-dim problems (even d=5), this takes a very long time."
def get_nquad(phi_func):
    # Define the integrand
    def integrand(*args):
        x = np.array(args)  # Convert positional arguments to a NumPy array
        norm_term = np.linalg.norm(Amat_np@x - ynp, 2)**2
        func_to_avg = phi_func(x)
        return func_to_avg * np.exp(-beta * (lam * np.sum(np.abs(x)) + 0.5 * norm_term))

    # Define integration limits for each dimension
    bounds = [(-np.inf, np.inf) for _ in range(n)]  # Adjust bounds as needed

    # Perform integration
    result, error = nquad(integrand, bounds)
    
    def integrand_for_scale(*args):
        x = np.array(args)
        norm_term = np.linalg.norm(Amat_np@x - ynp, 2)**2
        return np.exp(-beta * (lam*np.sum(np.abs(x)) + 0.5 * norm_term))
    
    Z_out, Z_err = nquad(integrand_for_scale, bounds)

    return result/Z_out, error, Z_err
    
T_burn = 800
T_samp = 1000
subsamp = 50

#tau_start = np.log10(0.6)
#tau_end = np.log10(0.01)
#tau_vals = np.logspace(tau_start,tau_end,12)

tau_start = 0.5
tau_end = np.log10(0.01)
tau_vals = np.logspace(np.log10(tau_start),tau_end,5)

#tau_start = np.log10(0.3)
#tau_end = np.log10(0.3)
#tau_vals = np.logspace(tau_start,tau_end,1)

#tau_vals = np.array([0.0019307 , 0.001])
burn_in_vals = ((2.5 * T_burn) // tau_vals).astype(int)
#niter_vals = ((2.5 * T_samp) // tau_vals).astype(int)
path_no = jnp.maximum(10000, (M // (tau_vals / tau_start)**(2)).astype(int))
#k_sub = (subsamp * (tau_start / tau_vals)).astype(int)
#setup = setup_data(jnp.array(Amat),n,lam,jnp.array(x0),jnp.array(y),M,gamma,beta,key)
#phi = lambda x: x.T @ x

save_xtra = os.path.join("SIMULATION","TOY","1D")
os.makedirs(save_xtra, exist_ok=True)

means = []
sum_burn_time = []
sum_samp_time = []

colors = plt.colormaps["tab10"].resampled(len(method_names)).colors

#save_runouts = []

for i in range(len(tau_vals)):    
    plt.figure()
    ax = plt.gca()
    
    tau = [tau_vals[i]] * len(method_names)
    sns.set()
    sns.set_palette("husl")
    
    
    ttl_xtra = rf"$d = {n}, tau = {tau[0]}"
    #save_xtra = f"toy_d_{n}_tau_{tau[0]}"
    #burn_in = int(5*T_burn / tau_vals[i])
    #burn_in = int(burn_in_vals[i])
    
    setup = setup_data(jnp.array(Amat),n,lam,jnp.array(x0),jnp.array(y),path_no[i],gamma,beta,key)
    
    #runout, burn_time, sample_time = Runner(setup, method_names).runner(niter_vals[i], tau, burn_in_vals[i], k_sub[i])
    runout, burn_time, sample_time = Runner(setup, method_names).runner(1, tau, burn_in_vals[i], 1)
    
    #save_runouts.append(runout)
    
    #runout, which_nans = exclude_nans(runout1)
    "Plot mean across particles to check mixing"
    min_ess_vals = []
    max_ess_vals = []
    mean_ess_vals = []
    plt.figure()
    for k in range(len(method_names)):
        plt.figure()
        plt.plot(np.mean(runout[k,:,0,:], axis = 1), label = method_names[k])
        plt.legend()
        ess_values = np.array([az.ess(runout[k,:,0,j]) for j in range(path_no[i])])
        print(f"Min ESS for {method_names[k]}: {ess_values.min():.2f}, Max ESS: {ess_values.max():.2f}")
        min_ess_vals.append(np.min(ess_values))
        max_ess_vals.append(np.max(ess_values))
        mean_ess_vals.append(np.mean(ess_values))


        
    x = np.arange(len(method_names))  # X positions for bars
    width = 0.3  # Width of the bars

    save_dist_str = "ESS_" + str(i) + ".pdf"
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars for min and max ESS
    ax.bar(x - width, min_ess_vals, width, label="Min ESS", color="blue", alpha=0.7)
    ax.bar(x , mean_ess_vals, width, label="Mean ESS", color="green", alpha=0.7)
    ax.bar(x + width, max_ess_vals, width, label="Max ESS", color="red", alpha=0.7)

    #add line for raw samples
    #raw_samples = niter_vals[i] // k_sub[i]  # Replace with the actual value of raw samples you want
    raw_samples = 1  # Replace with the actual value of raw samples you want
    ax.axhline(y=raw_samples, color='black', linestyle='--', label="Raw Samples", linewidth=1.5)

    # Labels and title
    ax.set_xlabel("Methods")
    ax.set_ylabel("ESS")
    ax.set_title("Min, Mean and Max ESS for Different Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)  # Set method names as x-axis labels
    ax.legend()
    #ax.set_ylim(0, niter // subsamp)
    fig_path_bar = os.path.join(save_xtra,save_dist_str)
    plt.savefig(fig_path_bar, format="pdf",bbox_inches="tight")
    plt.show()

    
    sum_burn_time.append(np.sum(burn_time, axis = 1))
    sum_samp_time.append(np.sum(sample_time, axis = 1))
    print('Starting mean calc')
    agg_means_phi = aggregate_mean(runout, phi)
    
    means.append(agg_means_phi)
    
    
    #you might need to adjust the plot range for the 'true' distr
    t= np.linspace(-3,3,10000) #plot range
    
    save_dist_str = "1d_cart_dist_tau_" + str(i) + ".pdf"

    fig_path1 = os.path.join(save_xtra,save_dist_str)
    plt.figure()
    for k in range(len(method_names)):
        sns.kdeplot(runout[k,-1,:,:].ravel(), color=colors[k], label=method_names[k])

    f = lambda x: np.abs(Amat[0]*x - y[0])**2/2 +lam*np.abs(x);
    z = np.exp(-beta*f(t))
    z = z/(np.sum(z)*(t[2]-t[1]))
    plt.plot(t,z, 'k--',label='true')
    plt.legend()
    plt.savefig(fig_path1, format="pdf",bbox_inches="tight")
    #files.download("1d_cart_dist.pdf")
    
    runout = None

means_arr = np.array(means)
sbt_arr = np.array(sum_burn_time)
sst_arr = np.array(sum_samp_time)

#mean_phi_hada = get_quasi_true_mean(phi)
#errs_quad = []
#for i in range(len(tau_vals)):
#    err_eula = np.abs(means_arr[i,0] - mean_phi_hada)
#    err_uvfb = np.abs(means_arr[i,1] - mean_phi_hada)
#    errs_quad.append([err_eula,err_uvfb])

#errs_arr = np.array(errs_quad)

#filename = "loglog_plot_quasi.pdf"
#fig_path1 = os.path.join(save_xtra,filename)


#fig_err,ax_err = plt.subplots()
#ax_err.loglog(tau_vals,errs_arr[:,0])
#ax_err.loglog(tau_vals,tau_vals)
#plt.loglog(tau_vals,errs_arr[:,0],label="EULA")
#plt.loglog(tau_vals,errs_arr[:,1],label="HADAMARD")
#plt.loglog(tau_vals,1.5*tau_vals, label= rf"Error = k \Delta t", linestyle='--')
#plt.legend()
#plt.title("Log-Log Plot - longtime hadamard error")
#plt.savefig(fig_path1, dpi=300, bbox_inches='tight')  # Save with high resolution
#plt.show



"Calculate integral numerically, using integrate.quad. For 1-D problems."
def get_quad(phi_func):
    # Define the integrand
    def integrand(*args):
        x = np.array(args)  # Convert positional arguments to a NumPy array
        norm_term = np.linalg.norm(Amat_np@[x] - ynp, 2)**2
        func_to_avg = phi_func(x)
        return func_to_avg * np.exp(-beta * (lam * np.sum(np.abs(x)) + 0.5 * norm_term))
    
    
    # Define integration limits for each dimension
    #bounds = (-np.inf, np.inf)  # Adjust bounds as needed

    # Perform integration
    result, error = quad(integrand, -np.inf, np.inf)
    
    def integrand_for_scale(*args):
        x = np.array(args)
        norm_term = np.linalg.norm(Amat_np@[x] - ynp, 2)**2
        return np.exp(-beta * (lam*np.abs(x) + 0.5 * norm_term))
    
    Z_out, Z_err = quad(integrand_for_scale, -np.inf, np.inf)

    return result/Z_out, error, Z_err
    

#mean_phi_quad = get_nquad(phi)
mean_phi_quad = get_nquad(phi_np)
errs_quad = []
for i in range(len(tau_vals)):
    err_quad_tmp = []
    for j in range(len(method_names)):
        err_quad_tmp.append(np.abs(means_arr[i,j] - mean_phi_quad[0]))
    errs_quad.append(err_quad_tmp)

errs_arr = np.array(errs_quad)

filename = "loglog_plot_quad.pdf"
fig_path2 = os.path.join(save_xtra,filename)

plt.figure()
for i in range(len(method_names)):
    plt.loglog(tau_vals,errs_arr[:,i],color=colors[i],label=method_names[i])
plt.loglog(tau_vals,tau_vals**2, label= rf"Error = dt^2", linestyle='--')
plt.loglog(tau_vals,tau_vals, label= rf"Error = dt", linestyle='--')
plt.loglog(tau_vals,tau_vals**(1/2), label= rf"Error = dt^0.5", linestyle='--')
plt.legend()
plt.title("Quadrature error vs stepsize")
plt.xlabel("Delta t")
plt.ylabel("Error")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show



max_sum_burn_t = np.max(sbt_arr)
mbt_ref = np.logspace(0.00001,max_sum_burn_t,len(tau_vals))


filename_time = "loglog_plot_quad_time.pdf"
fig_path3 = os.path.join(save_xtra,filename_time)

plt.figure()
for i in range(len(method_names)):
    plt.loglog(sbt_arr[:,i],errs_arr[:,i],color=colors[i],label=method_names[i])
#plt.loglog(mbt_ref,mbt_ref, label= rf"Error = O(Delta t)", linestyle='--')
#plt.loglog(mbt_ref,mbt_ref**(1/2), label= rf"Error = O(Delta t^(0.5))", linestyle='--')
plt.legend()
plt.title("Quadrature error vs burn-in time")
plt.xlabel("T")
plt.ylabel("Error")
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show


"Create a bar-chart of mean per-iteration times"
filename_bar = "times_bar.pdf"
fig_path4 = os.path.join(save_xtra,filename_bar)
time_values = np.mean(sbt_arr / burn_in_vals[:,np.newaxis], axis = 0)
'Create a value-dependent colour scheme'
norm = plt.Normalize(min(time_values), max(time_values))
color_bar = sns.color_palette("Blues", as_cmap=True)(norm(time_values))

sns.barplot(x=method_names, y=time_values, palette=color_bar)

plt.xlabel("Numerical Method")
plt.ylabel("t(s)")
plt.title("Mean time per iteration, per stepsize")
plt.savefig(fig_path4, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


def get_mcmc_mh(phi_func, num_samp, burnin_mc, thinning_mc):
    def log_density(x):
        norm_term = np.linalg.norm(Amat_np@x - ynp, 2)**2
        return -beta * (lam * np.sum(np.abs(x)) + 0.5 * norm_term)

    def metropolis_hastings(num_samples, initial_x, proposal_std, burnin_mc, thinning_mc):
        samples = []
        x_current = initial_x
        for i in range(burnin_mc):
            if i % 5000 == 0:
                print('MCMC burn-in iteration' + str(i))
            # Propose a new sample
            x_proposal = x_current + np.random.normal(0, proposal_std, size=x_current.shape)
        
            # Compute acceptance ratio
            log_p_current = log_density(x_current)
            log_p_proposal = log_density(x_proposal)
            acceptance_ratio = np.exp(log_p_proposal - log_p_current)
        
            # Accept or reject
            if np.random.rand() < acceptance_ratio:
                x_current = x_proposal
        for i in range(num_samples):
            if i % 5000 == 0:
                print('MCMC iteration' + str(i))
            # Propose a new sample
            x_proposal = x_current + np.random.normal(0, proposal_std, size=x_current.shape)
        
            # Compute acceptance ratio
            log_p_current = log_density(x_current)
            log_p_proposal = log_density(x_proposal)
            acceptance_ratio = np.exp(log_p_proposal - log_p_current)
        
            # Accept or reject
            if np.random.rand() < acceptance_ratio:
                x_current = x_proposal
                
            # Record samples dictated by thinning factor
            if i % thinning_mc == 0:
                samples.append(x_current)
    
        return np.array(samples)

    # Parameters (example)
    proposal_std = 0.1
    initial_x = np.zeros(n)

    # Running MCMC
    samples = metropolis_hastings(num_samp, initial_x, proposal_std, burnin_mc, thinning_mc)
    expected_value = np.mean([phi_func(x) for x in samples])
    return expected_value


    

#for i in range(no_methods):
#    plt.stem(jnp.mean(runout[i],axis=0))
    
'x_arr is a vector of last iterates of size (no. algorithms, dimension of iterate, no. particles)'


    

