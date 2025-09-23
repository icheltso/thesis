# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:53:39 2024

@author: ichel
"""

"Tool for running various algorithms given parameters such as stepsize, niter, lambda."
"Returns mean, std, final iterate as concatenated arrays for all chosen algorithms."

import os
os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from helpers import Helper
from algo import Solver

import jax.numpy as jnp
from jax import random
from jax import jit

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

        #record n samples
        subsamp_no = n // subsamp
        samples = np.zeros((subsamp_no,dim,part))
        for j in range(part):
            print(f'Started for particle {j}')
            x = init[:,j]
            
            #x = init[:,j].reshape(-1,1)
            for b in range(burn_in):
                x = Iterate(x)
                if b % 5000 == 0:
                    print(f'Burn-in iteration {b}')
            for i in range(n):
                x = Iterate(x)
                if (i+1)%subsamp==0:
                    samples[i//subsamp,:,j] = x.reshape(-1)
            
        return samples
    
    def compute_time_averages(self,iterates):
    
        niter, dim = iterates.shape
        time_avg = np.zeros((niter, dim))
    
        # Initialize the first time average for each dimension
        time_avg[0] = iterates[0]
    
        # Compute time averages iteratively for each dimension
        for n in range(1, niter):
            time_avg[n] = (n * time_avg[n-1] + iterates[n]) / (n + 1)
    
        return time_avg
        

    def runner(self, niter, tau, burn_in, subsamp):
        "Subsamp is the iteration difference two subsampled particles. E.g., at subsamp = 10, we subsample every 10th particle. At subsamp = 1, we sample every particle. "
        x_dim = np.size(self.initx,0)
        print(x_dim)
        x_part = np.size(self.initx,1)
        no_algs = len(self.alg_id)
        subsamp_no = niter // subsamp
        "Store a designated subsampled chain of x_part parallel particles."
        subsamp_seq = np.zeros((no_algs,subsamp_no,x_dim,x_part))
        for i in range(no_algs):
            print(f"Starting method {self.alg_id[i]}")
            method = getattr(self.algo, self.alg_id[i])
            metadata = getattr(method, '_metadata', {})
            "Initialize variables depending on type of method (uv reparam or none)"
            if metadata.get('is_uv') == 1:
                x = jnp.concatenate((jnp.abs(self.initx),self.initv))
                #x_first =  np.zeros((niter, 2*x_dim))
            elif metadata.get('is_uv') == 0:
                x = self.initx
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
                samples_all_part = self.generate_samples_x(Iterate, xnp, niter, burn_in, subsamp)
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
                    samples_all_part = []
                    for jb in range(burn_in):
                        if jb % 5000 == 0:
                            print('Burn-in iteration' + str(jb))
                        x,mean,std,key = method_jit(x,tau[i],self.lam,self.gamma,key)
                        #print(x.shape)
                        #print(subsamp_seq.shape)
                    print('Finished Burn-in.')
                    for j in range(niter):
                        if j % 5000 == 0:
                            print('Convergence iteration' + str(j))
                        x,mean,std,key = method_jit(x,tau[i],self.lam,self.gamma,key)
                        if (j+1)%subsamp == 0:
                            #samples_all_part[j//subsamp,:,:] = x
                            samples_all_part.append(x)

                    samples_all_part = np.array(samples_all_part)
                    if metadata.get('is_uv') == 1:
                        subsamp_seq[i] = samples_all_part[:,:x_dim,:] * samples_all_part[:,x_dim:,:]
                    elif metadata.get('is_uv') == 0:
                        subsamp_seq[i] = samples_all_part
                    else:
                        subsamp_seq[i] = samples_all_part[:,:x_dim,:]
                        
        #print(subsamp_seq.shape)
        return subsamp_seq
