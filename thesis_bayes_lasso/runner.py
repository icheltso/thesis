# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:53:39 2024

@author: ichel
"""

"Tool for running various algorithms given parameters such as stepsize, niter, lambda."
"Returns mean, std, final iterate as concatenated arrays for all chosen algorithms."

import os
import time
os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from helpers import Helper
from algo import Solver

import jax.numpy as jnp
from jax import random
from jax import jit

import matplotlib.pyplot as plt

#soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)
#Burn-in
#burn_in = 1000



class Runner:
    def __init__(self, setup, alg_id):
        datadict = setup.get_data()
        self.gamma = datadict['gamma']
        self.lam = datadict['lam']
        self.initx = datadict["initx"]
        self.initv = datadict["initv"]
        self.initz = datadict["initz"]
        'alg_id should be list of names of algorithms that you want to test for'
        self.alg_id = alg_id
        self.algo = Solver(setup)
        self.helpr = Helper(setup)
        
    "To be used with non-jax method."
    def generate_samples_x(self, Iterate, init, n, burn_in, subsamp):
        x = init
        #x = init.reshape(-1,1)

        #burn in 
        #for i in range(burn_in):
        #    x = Iterate(x)
        dim = np.size(x,0)
        part = np.size(x,1)
        
        burn_time = np.zeros(burn_in)
        samp_time = np.zeros(n)

        #record n samples
        subsamp_no = n // subsamp
        samples = np.zeros((subsamp_no,dim,part))
        for j in range(part):
            print(f'Started for particle {j}')
            x = init[:,j]
            
            #x = init[:,j].reshape(-1,1)
            for b in range(burn_in):
                start = time.perf_counter()
                x = Iterate(x)
                end = time.perf_counter()
                burn_time[b] = end - start
                if b % 1000 == 0:
                    print(f'Burn-in iteration {b}')
            for i in range(n):
                start = time.perf_counter()
                x = Iterate(x)
                end = time.perf_counter()
                samp_time[i] = end - start
                if i % 1000 == 0:
                    print(f'Sampling iteration {i}')
                if (i+1)%subsamp==0:
                    samples[i//subsamp,:,j] = x.reshape(-1)
                    #print(f'Sampling iteration {i}')
            
        return samples, burn_time, samp_time
    
    def compute_time_averages(self, iterates, method_name):
        '''Compute time averages of phi = x^2'''
    
        niter, dim, npaths = iterates.shape
        norm_sq = np.sum(iterates**2, axis=1)  # shape: (niter, npaths)

        time_avg_per_path = np.zeros_like(norm_sq)  # (niter, npaths)
        time_avg_per_path[0] = norm_sq[0]
    
        for n in range(1, niter):
            time_avg_per_path[n] = (n * time_avg_per_path[n-1] + norm_sq[n]) / (n + 1)
    
        # Average over paths
        mean_time_avg = np.mean(time_avg_per_path, axis=1)  # shape: (niter,)
        
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, niter + 1), mean_time_avg, label=r"$\mathbb{E}[\|x\|^2]$")
        plt.xlabel("Iteration")
        plt.ylabel(r"Time average of $\|x\|^2$")
        plt.title(f"Time Average of $\\|x\\|^2$ during Burn-in" + (f" ({method_name})"))
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
        return mean_time_avg
        

    def runner(self, niter, tau, burn_in, subsamp):
        "Subsamp is the iteration difference two subsampled particles. E.g., at subsamp = 10, we subsample every 10th particle. At subsamp = 1, we sample every particle. "
        x_dim = np.size(self.initx,0)
        print(x_dim)
        x_part = np.size(self.initx,1)
        no_algs = len(self.alg_id)
        subsamp_no = niter // subsamp
        "Store a designated subsampled chain of x_part parallel particles."
        subsamp_seq = np.zeros((no_algs,subsamp_no,x_dim,x_part))
        #burn_in_seq = np.zeros((burn_in,x_dim,x_part))
        burn_in_time = np.zeros((no_algs,burn_in))
        sampling_time = np.zeros((no_algs,niter))
        for i in range(no_algs):
            print(f"Starting method {self.alg_id[i]}")
            method = getattr(self.algo, self.alg_id[i])
            metadata = getattr(method, '_metadata', {})
            "Initialize variables depending on type of method (uv reparam or none or cartesian)"
            if metadata.get('is_uv') == 1:
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
                #x_first =  np.zeros((niter, 2*x_dim))
            elif metadata.get('is_uv') == 0:
                x = self.initx
                #x_first =  np.zeros((niter, x_dim))
            elif metadata.get('is_uv') == 3:
                x = jnp.concatenate((self.initx,self.initv,self.initz))
                    #x_first =  np.zeros((niter, x_dim))
            else:
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
                #x_first =  np.zeros((niter, 2*x_dim))
                
            key = random.key(0)
            
            "Whether this is a numpy-based method that simulates particles individually."
            if metadata.get('all_part') == False:
                xnp = np.array(x)
                Iterate = lambda z: method(z,tau[i],self.lam,self.gamma)
                "Start with burn-in"
                samples_all_part, burn_in_time[i,:], sampling_time[i,:] = self.generate_samples_x(Iterate, xnp, niter, burn_in, subsamp)
                if metadata.get('is_uv') == 1:
                    subsamp_seq[i] = samples_all_part[:,:x_dim,:] * samples_all_part[:,x_dim:,:]
                elif metadata.get('is_uv') == 0:
                    subsamp_seq[i] = samples_all_part
                else:
                    subsamp_seq[i] = samples_all_part[:,:x_dim,:]
            else: 
                "If method simulates multiple parallel particles"
                "If method is multiple-timestep, give niter and omit x"
                if metadata.get('one_timestep') == False:
                    dummy, mean_dummy = method(tau[i],self.lam,self.gamma,key,niter,burn_in)
                else: 
                    "If method is multi-particle, and simulates one step at a time. All such methods use jax/jit."
                    method_jit = jit(method)
                    #method_jit = method
                    #if metadata.get('is_uv') == 0:
                    #    samples_all_part = np.zeros((subsamp_no,x_dim,x_part))
                    #else:
                    for jb in range(burn_in):
                        if jb % 5000 == 0:
                            print('Burn-in iteration' + str(jb))
                        start = time.perf_counter()
                        x,mean,std,key = method_jit(x,tau[i],self.lam,self.gamma,key)
                        end = time.perf_counter()
                        burn_in_time[i,jb] = end - start
                        
                        #Unhash code below for time averages during burn-in
                        
                        #if metadata.get('is_uv') == 1:
                        #    burn_in_seq[jb] = x[:x_dim,:] * x[x_dim:,:]
                        #elif metadata.get('is_uv') == 0:
                        #    burn_in_seq[jb] = x
                        #elif metadata.get('is_uv') == 3:
                        #    burn_in_seq[jb] = self.helpr.u_sq(x[:x_dim,:], x[x_dim:2*x_dim,:]) * x[2*x_dim:,:]
                        #else:
                        #    burn_in_seq[jb] = x[:x_dim,:]
                        
                        #print(x)
                        #time.sleep(1)
                        #print(x.shape)
                        #print(subsamp_seq.shape)
                    print('Finished Burn-in.')
                    #self.compute_time_averages(burn_in_seq, method)
                    for j in range(niter):
                        if j % 5000 == 0:
                            print('Convergence iteration' + str(j))
                        start = time.perf_counter()
                        x,mean,std,key = method_jit(x,tau[i],self.lam,self.gamma,key)
                        end = time.perf_counter()
                        sampling_time[i,j] = end - start
                        #print(x)
                        #time.sleep(1)
                        if (j+1)%subsamp == 0:
                            if metadata.get('is_uv') == 1:
                                subsamp_seq[i,(j) // subsamp] = x[:x_dim,:] * x[x_dim:,:]
                            elif metadata.get('is_uv') == 0:
                                subsamp_seq[i,(j) // subsamp] = x
                            elif metadata.get('is_uv') == 3:
                                subsamp_seq[i,(j) // subsamp] = self.helpr.u_sq(x[:x_dim,:], x[x_dim:2*x_dim,:]) * x[2*x_dim:,:]
                            else:
                                subsamp_seq[i,(j) // subsamp] = x[:x_dim,:]
                            
                            
                    
                        
        #print(subsamp_seq.shape)
        return subsamp_seq, burn_in_time, sampling_time
