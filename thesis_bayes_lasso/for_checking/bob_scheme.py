# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:13:22 2025

@author: ichel
"""
"Playing with B-O-B scheme for cartesian parametrisation to check for instabilities and bugs."

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
    #print(initx)
    #print(initv)
    #print(initz)
    
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



niter = 10**6
#burn_in = 10**5
subsamp = 10
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

#M = 10**4# number of particles
M = 1
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
Amat = [1]
x0 = jnp.zeros((n,))
x0 = x0.at[n//4].set(10)
x0 = x0.at[n//2].set(3)
#y = Amat@x0
y=[3]

#lam = (1/2)*jnp.max(jnp.abs(Amat.T@y))
lam = 2.7

#parameters for prox
Lf = np.linalg.norm(Amat)**2
gamma = 1 / (5*Lf)
tau_prox = 1 / (Lf + 1/gamma)

#tau = [tau_prox,
#       tau_hada
#       ]

#phi = lambda x: np.linalg.norm(x,2)**2
phi = lambda x: jnp.sum(x**2)

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
    
T_burn = 10000

#tau_start = np.log10(0.3)
#tau_start = np.log10(0.01)
#tau_end = np.log10(0.01)
#tau_vals = np.logspace(tau_start,tau_end,10)

tau = [0.3]
#burn_in = int(T_burn / tau[0])
burn_in = 0

#tau_start = np.log10(0.01)
#tau_end = np.log10(0.01)
#tau_vals = np.logspace(tau_start,tau_end,1)

#tau_vals = np.array([0.0019307 , 0.001])
burn_in_vals = T_burn / tau[0]
setup = setup_data(jnp.array(Amat),n,lam,jnp.array(x0),jnp.array(y),M,gamma,beta,key)
#phi = lambda x: x.T @ x

save_xtra = os.path.join("SIMULATION","TOY","1D")
os.makedirs(save_xtra, exist_ok=True)

method_names = ['cart_BOB',
                #'cart_EM',
                #'cart_seq',
                #'uv_FB'
                ]
runout = Runner(setup, method_names).runner(niter, tau, burn_in, subsamp) 
print('Starting mean calc')
agg_means_phi = aggregate_mean(runout, phi)
expected_value_bob = agg_means_phi[0]

mean_phi_quad = get_nquad(phi)
err_bob = np.abs(expected_value_bob - mean_phi_quad[0])
rel_err_bob = err_bob / mean_phi_quad[0]

print(rel_err_bob)

filename = "loglog_plot_quasi.pdf"
fig_path1 = os.path.join(save_xtra,filename)





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


    

