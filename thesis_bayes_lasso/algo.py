# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:49:53 2023

@author: ichel
"""

"One step of various algorithms"

import os
import numpy as np
from numpy.linalg import inv
#import importlib
from scipy.stats import invgamma, norm,invgauss

import matplotlib.pyplot as plt
from helpers import Helper

import jax.numpy as jnp
from jax import random
from jax import jit
from jax import vmap
import time

soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)

"metadata legend:" 
"is_uv = 0 - Method doesn't use x = u odot v reparametrization"
"is_uv = 1 - Method uses x = u odot v reparametrization"
"is_uv = 2 - Method doesn't use uv reparam, but requires tracking of additional auxiliary variables (Gibbs), stored within the iterate."
"one_timestep = True - Method only iterates through a single timestep"
"one_timestep = False - Method iterates through entire time Delta*niter in one go."
"all_part = True - Method simulates p particles simultaneously. Typically, the iterate is a vector of dimension (d x p) or (2d x p), where d is the dimension of the problem."
"all_part = False - Method only simulates a single particle. Useful for methods that are difficult to vectorize due to limitations of JAX, or for other reasons. Normally use numpy in this case."

def method_metadata(**metadata):
    def decorator(func):
        func._metadata = metadata
        return func
    return decorator

class Solver:
    def __init__(self, setup):
        current_directory = os.getcwd()
        #file_path = os.path.join(current_directory, "data.npz")
        #file_path = os.path.join(current_directory, "data.pkl")
        #with open(file_path, 'rb') as file:
        #    loaded_dump = dill.load(file)
        #data = np.load(file_path)
        #self.A = loaded_dump["A"]
        #self.As = loaded_dump["As"]
        'Pass A, and adjoint explicitly, as jax doesnt support serialising functions.'
        datadict = setup.get_data()
        self.A = setup.A
        self.As = setup.As
        self.n = datadict["n"]
        self.M = datadict["M"]
        self.y = datadict["y"]
        self.y2 = datadict["y2"]
        self.R = datadict["R"]
        self.sigma = datadict["sigma"]
        
        
        #self.n = loaded_dump["n"]
        #self.M = loaded_dump["M"]
        #self.y = jnp.array(loaded_dump["y"])
        #self.y2= jnp.array(loaded_dump["y2"])
        
        #current_directory = os.getcwd()
        #file_path = os.path.join(current_directory, "data.npz")

        #data = np.load(file_path)
        #self.A = data["A"]
        #self.n = data["n"]
        #self.M = data["M"]
        #self.y = data["y"]
        #self.y2= data["y2"]
        self.helpr = Helper(setup)
    
    
    @method_metadata(description='MYULA - https://arxiv.org/pdf/1306.0187', is_uv=0, one_timestep=True, all_part = True)
    def EULA(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        #noise=random.normal(key,[self.n,self.M]) * self.sigma*jnp.sqrt(tau)
        noise=random.normal(subkey,[len(x[:,0]),self.M]) * self.sigma*jnp.sqrt(tau)
        print(noise.shape)
        x = x - tau*self.helpr.grad_pula(x,lam,gamma) + noise
        #print(np.count_nonzero(np.isnan(x)))
        mean = jnp.mean(x,1)
        std = jnp.std(x,1)
        return x,mean,std,key
    
    @method_metadata(description='Proximal MCMC - https://arxiv.org/pdf/1306.0187', is_uv=0, one_timestep=True, all_part = True)
    def PROXL1(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        #noise=random.normal(key,[self.n,self.M]) * self.sigma*jnp.sqrt(tau)
        noise=random.normal(key,[len(x[:,0]),self.M]) * self.sigma*jnp.sqrt(tau)
        x = soft( x - tau* self.helpr.dF(x) + noise, tau*lam )
        mean = jnp.mean(x,1)
        std = jnp.std(x,1)
        return x,mean,std,key
    
    @method_metadata(description='Proximal MCMC - https://arxiv.org/pdf/1306.0187', is_uv=0, one_timestep=True, all_part = False)
    def one_part_eula(self, x,  tau, lam, gamma):
        noise = np.random.randn(*x.shape) *self.sigma*np.sqrt(tau)
        #print('hi')
        #print(x.shape)
        x = x - tau*self.helpr.grad_pula_1d(x,lam,gamma) + noise
        
        return x
    
    
    @method_metadata(description='FB envelope https://arxiv.org/pdf/2201.09096', is_uv=0, one_timestep=True, all_part = True)
    def EULA_FB(self, x,  tau, lam, gamma, key ):
        key, subkey = random.split(key)
        noise=random.normal(key,[self.n,self.M])*self.sigma*jnp.sqrt(tau)

        x = x - tau*self.helpr.gradFB(x,lam,gamma) + noise

        mean = jnp.mean(x,1)
        std = jnp.std(x,1)
        return x,mean,std,key
    
    
    # Gibbs Sampler for Bayesian Lasso with sigma^2 = 1
    "Numpy-Gibbs - one particle"
    @method_metadata(description='Gibbs Sampler', is_uv=2, one_timestep=True, all_part = False)
    def one_part_gibbs(self, x_eta,  tau, lam, gamma):
        beta = 2 / self.sigma**2
    
        #n,p = self.A.shape
        
        eta = x_eta[self.n:]

        # Sample x | y, X, eta
        V_x = inv(beta*self.As(self.A(np.identity(self.n))) + np.diag(1/eta))
        m_x = beta*V_x @ self.As(self.y)
        x = np.random.multivariate_normal(m_x, V_x)

        # Sample eta_j | x_j
        for j in range(self.n):
            eta[j]= 1/invgauss.rvs(mu=abs(1./(beta*lam*np.abs(x[j]))), scale=(lam*beta)**2)

    
        return np.concatenate((x,eta))
    
    
    #Implicit-Drift Algorithm
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf',
                     is_uv=1, one_timestep=True, all_part = True)
    def uv_FB(self, x,  tau, lam, gamma, key ):
        beta = 2 / self.sigma**2
        #self.one_step_uv_FB.is_uv = True
        S1 = lambda xt: (xt+jnp.sqrt(xt**2 + 4*tau*(1+tau*lam)/beta))/2
        key, subkey = random.split(key)
        #noise= random.normal(key,[2*self.n,self.M]) *self.sigma*jnp.sqrt(tau)
        #print(len(x[:,0]))
        noise= random.normal(key,[len(x[:,0]),self.M]) *self.sigma*jnp.sqrt(tau)
        #print(noise)
        
        z = x - tau*self.helpr.Grd(x) + noise
        #print(z)
        #print(np.count_nonzero(np.isnan(z)))
        #print(z)
        z = z.at[:self.n,:].set(S1(z[:self.n,:]))
        x = z/(1+tau*lam)
        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std,key
    
    #Implicit-Drift Algorithm
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf',
                     is_uv=1, one_timestep=True, all_part = True)
    def uv_FB_trunc(self, x,  tau, lam, gamma, key):
        #self.one_step_uv_FB.is_uv = True
        beta = 2 / self.sigma**2
        S1 = lambda xt: (xt+jnp.sqrt(xt**2 + 4*tau*(1+tau*lam)/beta))/2


        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M]) *self.sigma*jnp.sqrt(tau)
        colnorms = jnp.linalg.norm(x, ord=2, axis=0) < self.R
        z = jnp.where(colnorms  ,  x - tau*self.helpr.Grd(x) + noise  ,  x + noise)
        #z = x - tau*self.helpr.Grd(x) + noise
        z = z.at[:self.n,:].set(S1(z[:self.n,:]))
        x = z/(1+tau*lam)
        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std,key
    
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf', is_uv=1, one_timestep=True, all_part = False)
    def one_part_uvfb(self, xnp,  tau, lam, gamma):
        #print(xnp.shape)
        #S1np = lambda xt: (xt+np.sqrt(xt**2 + 4*tau*(1+tau*lam)))/2
        def S1(x):
            beta = 2 / self.sigma**2
            x = np.abs(x)
            if x<1:
                return (x+np.sqrt(x**2 + 4*tau*(1+tau*lam)/beta))/2.
            else:
                return 0.5*x+0.5*x* np.sqrt(1+ (4/beta*tau*(1+tau*lam)/x)/x)
        S1 = np.vectorize(S1)
        noise = np.random.randn(*xnp.shape)*self.sigma*np.sqrt(tau)
        z = xnp - tau*self.helpr.Grd_1d(xnp) + noise
        z[:self.n] = S1(z[:self.n])
        #print("Max val: " + str(np.max(z)))
        #print("Min u - " + str(np.min(z[:self.n])))
        x = z/(1+tau*lam)

        return x
    
    @method_metadata(description='Based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf', is_uv=1, one_timestep=True, all_part = False)
    def one_part_uvfb_trunc(self, xnp,  tau, lam, gamma, R = 100):
        #print(xnp.shape)
        #S1np = lambda xt: (xt+np.sqrt(xt**2 + 4*tau*(1+tau*lam)))/2
        def S1(x):
            beta = 2 / self.sigma**2
            x = np.abs(x)
            if x<1:
                return (x+np.sqrt(x**2 + 4*tau*(1+tau*lam)/beta))/2.
            else:
                return 0.5*x+0.5*x* np.sqrt(1+ (4/beta*tau*(1+tau*lam)/x)/x)
        S1 = np.vectorize(S1)
        noise = np.random.randn(*xnp.shape)*self.sigma*np.sqrt(tau)
        if np.linalg.norm(xnp,2) < R:
            z = xnp - tau*self.helpr.Grd_1d(xnp) + noise
        else:
            z = xnp + noise
        z[:self.n] = S1(z[:self.n])
        print("Max val: " + str(np.max(z)))
        print("Min u: " + str(np.min(z[:self.n])))
        x = z/(1+tau*lam)

        return x
    
    @method_metadata(description='Cartesian Parametrization, method based on BAOAB scheme', is_uv=3, one_timestep=True, all_part = True)
    def cart_BOB(self, x,  tau, lam, gamma, key):
        beta = 2 / self.sigma**2
        key, subkey = random.split(key)
        noise= random.normal(key,[3*self.n,self.M])
        #print(len(x[:,0]))
        #noise= random.normal(key,[len(x[:,0]),self.M]) *self.sigma*jnp.sqrt(tau)
        
        
        """Solve B (Euler step)"""
        x = x + self.helpr.Grd_3(x) * tau/2
        """Solve O (OU process)"""
        x = x*jnp.exp(-lam * tau) + jnp.sqrt((1 / (beta*lam)) * (1-jnp.exp(-2*lam*tau)) ) * noise
        """Solve B (Euler step)"""
        x = x + self.helpr.Grd_3(x) * tau/2
        
        x_real = self.helpr.u_sq(x[:self.n,:], x[self.n:2*self.n,:]) * x[2*self.n:,:]
        #print("x = " + str(x_real))
        
        mean = jnp.mean(x_real,1)
        std = jnp.std(x_real,1)
        
        return x, mean, std, key
    
    @method_metadata(description='Cartesian Parametrization, method based on BAOAB scheme, reduced gradient computation'
                     , is_uv=3, one_timestep=True, all_part = True)
    def cart_OBO(self, x,  tau, lam, gamma, key):
        beta = 2 / self.sigma**2
        
        
        newkey,key1,key2 = random.split(key, num=3)
        noise1=random.normal(key1,[3*self.n,self.M])
        noise2=random.normal(key2,[3*self.n,self.M])
        
        
        """Solve O (OU process)"""
        x = x*jnp.exp(-lam * tau/2) + jnp.sqrt((1 / (beta*lam)) * (1-jnp.exp(-2*lam*tau/2)) ) * noise1
        """Solve B (Euler step)"""
        x = x + self.helpr.Grd_3(x) * tau
        """Solve B (Euler step)"""
        x = x*jnp.exp(-lam * tau/2) + jnp.sqrt((1 / (beta*lam)) * (1-jnp.exp(-2*lam*tau/2)) ) * noise2
        
        x_real = self.helpr.u_sq(x[:self.n,:], x[self.n:2*self.n,:]) * x[2*self.n:,:]
        #print("x = " + str(x_real))
        
        mean = jnp.mean(x_real,1)
        std = jnp.std(x_real,1)
        
        return x, mean, std, newkey
    
    @method_metadata(description='Cartesian Parametrization, Sequential Splitting', is_uv=3, one_timestep=True, all_part = True)
    def cart_seq(self, x,  tau, lam, gamma, key):
        beta = 2 / self.sigma**2
        key, subkey = random.split(key)
        noise= random.normal(key,[3*self.n,self.M])
        
        """Solve B (Euler step)"""
        x = x + self.helpr.Grd_3(x) * tau
        """Solve O (OU process)"""
        x = x*jnp.exp(-lam * tau) + jnp.sqrt((1 / (beta*lam)) * (1-jnp.exp(-2*lam*tau)) ) * noise

        
        x_real = self.helpr.u_sq(x[:self.n,:], x[self.n:2*self.n,:]) * x[2*self.n:,:]
        #print("x = " + str(x_real))
        
        mean = jnp.mean(x_real,1)
        std = jnp.std(x_real,1)
        
        return x, mean, std, key
    
    @method_metadata(description='Cartesian Parametrization, Euler-Maruyama', is_uv=3, one_timestep=True, all_part = True)
    def cart_EM(self, x,  tau, lam, gamma, key):
        key, subkey = random.split(key)
        noise= random.normal(key,[3*self.n,self.M])
        
        
        """Solve (Euler step)"""
        x = x - lam * x * tau + self.helpr.Grd_3(x) * tau + jnp.sqrt(tau) * self.sigma * noise
        
        x_real = self.helpr.u_sq(x[:self.n,:], x[self.n:2*self.n,:]) * x[2*self.n:,:]
        #print("x = " + str(x_real))
        
        mean = jnp.mean(x_real,1)
        std = jnp.std(x_real,1)
        
        return x, mean, std, key
    
    @method_metadata(description='Cartesian Parametrization, Pseudo-Euler-Maruyama, based on scheme in Leimkuhler paper',
                     is_uv=3, one_timestep=True, all_part = True)
    def cart_PEM(self, x,  tau, lam, gamma, key, R = jnp.inf):
        noise_old = random.normal(key,[3*self.n,self.M]) 
        subkey1, subkey2 = random.split(key)
        noise_new = random.normal(subkey1,[3*self.n,self.M])
        
        
        
        """Solve (Pseudo-Euler step)"""
        x = x - lam * x * tau + self.helpr.Grd_3(x) * tau + jnp.sqrt(tau/4) * self.sigma * (noise_old + noise_new)
        
        grd = self.helpr.Grd_3(x)
        colnorms = jnp.linalg.norm(grd, ord=2, axis=0) < self.R
        x = jnp.where(colnorms, 
                      x - lam * x * tau + grd * tau + jnp.sqrt(tau/4) * self.sigma * (noise_old + noise_new),  
                      x - lam * x * tau + jnp.sqrt(tau/4) * self.sigma * (noise_old + noise_new))
        
        
        x_real = self.helpr.u_sq(x[:self.n,:], x[self.n:2*self.n,:]) * x[2*self.n:,:]
        #print("x = " + str(x_real))
        
        mean = jnp.mean(x_real,1)
        std = jnp.std(x_real,1)
        
        return x, mean, std, subkey1
    
    # Bessel scheme
    @method_metadata(description='Bessel Split',
                     is_uv=1, one_timestep=True, all_part = True)
    def uv_Bessel(self, x,  tau, lam, gamma, key ):
        key1,key2 = random.split(key)
        noise1=random.normal(key1,[self.n,self.M])*self.sigma*jnp.sqrt(tau)
        key2a,key2b = random.split(key2)
        noise2a=random.normal(key2a,[self.n,self.M])*self.sigma*jnp.sqrt(tau)
        noise2b=random.normal(key2b,[self.n,self.M])*self.sigma*jnp.sqrt(tau)

        #forward Euler step
        x = x - tau*self.helpr.Grd(x)

        #Resolve Bessel directly
        x = x.at[self.n:,:].set( x[self.n:,:] + noise1)
        # update u
        x=x.at[:self.n,:].set( jnp.sqrt((x[:self.n,:] + noise2a)**2 + noise2b**2))

        uv = self.helpr.ru(x)*self.helpr.rv(x)
        mean = jnp.mean(uv,1)
        std = jnp.std(uv,1)
        return x, mean, std, key1
    
    @method_metadata(description='Bessel Split', is_uv=1, one_timestep=True, all_part = False)
    def one_part_bessel(self, x,  tau, lam, gamma):
        #noise1 = np.random.randn(self.n,1) *np.sqrt(2*tau)
        #noise2a=np.random.randn(self.n,1) *np.sqrt(2*tau)
        #noise2b=np.random.randn(self.n,1) *np.sqrt(2*tau)
        
        noise1 = np.random.randn(*x.shape) *self.sigma*np.sqrt(tau)
        noise1 = noise1[self.n:]
        
        noise2 = np.random.randn(*x.shape)*self.sigma*np.sqrt(tau)
        noise2a= noise2[self.n:]
        noise2b= noise2[:self.n]

        #forward Euler step
        x = x - tau*self.helpr.Grd_1d(x)

        #Resolve Bessel directly
        #x = x.at[self.n:,:].set( x[self.n:,:] + noise1)
        #x[self.n:,:] = x[self.n:,:] + noise1
        x[self.n:] = x[self.n:] + noise1
        # update u
        #x[:self.n,:] = np.sqrt((x[:self.n,:] + noise2a)**2 + noise2b**2)
        x[:self.n] = np.sqrt((x[:self.n] + noise2a)**2 + noise2b**2)
        
        return x
    
    @method_metadata(description='Metropolis-Adjusted algorithm, based on CIR scheme - https://www.uni-muenster.de/Stochastik/dereich/Publikationen/Preprints/cir.pdf',
                     is_uv=1, one_timestep=True, all_part = True)
    def MASCIR(self, x,  tau, lam, gamma, key ):
        beta = 2 / self.sigma**2
        key, subkey = random.split(key)
        noise= random.normal(key,[2*self.n,self.M]) *self.sigma*jnp.sqrt(tau)
        b = 1 + tau*lam
        "Update uv-CIR as in the HADAMARD-MALA subsection in the 'related methods' section of the paper"
        z = x - tau*self.helpr.Grd(x) + noise
        Puz = self.helpr.ru(z)
        Pvz = self.helpr.rv(z)
        Puz = (Puz + jnp.sqrt(Puz**2 + 4*tau*b/beta)) / (2*b)
        Pvz = Pvz / b
        Pz = jnp.concatenate((Puz,Pvz))
        
        
        "Obtain target densities"
        tdensx = self.helpr.get_dens_lsqr_unscale(x, lam)
        tdensx2 = self.helpr.get_dens_lsqr_unscale(Pz, lam)
        "Obtain transition densities as in the HADAMARD-MALA subsection in the 'related methods' section of the paper"
        qx = self.helpr.q_CIR(x, Pz, tau, lam)
        qx2 = self.helpr.q_CIR(Pz, x, tau, lam)
        
        
        
        alpha = jnp.minimum(jnp.ones(self.M), (tdensx2 * qx)/(tdensx * qx2))
        finx = jnp.zeros((2*self.n,self.M))
        unifvec = random.uniform(subkey, shape=(self.M,), minval=0.0, maxval=1.0)
        comparr = unifvec <= alpha
        
        #print(comparr)
        
        finx = jnp.where(comparr,Pz,x)
        #finx = jnp.where(comparr,x,Pz)
        finuv = self.helpr.ru(finx)*self.helpr.rv(finx)
        mean = jnp.mean(finuv,1)
        std = jnp.std(finuv,1)
        return finx, mean, std,key
    
    "Numpy-MALA - one particle"
    @method_metadata(description='MALA', is_uv=1, one_timestep=True, all_part = False)
    def one_part_mala(self, x,  tau, lam, gamma):
        "Update uv-CIR as in the HADAMARD-MALA subsection in the 'related methods' section of the paper"
        Pz = self.one_part_uvfb(x, tau, lam, gamma)
        #print(Pz)
        "Obtain target densities"
        tdensx = self.helpr.get_dens_lsqr_unscale_1p(x, lam)
        tdensx2 = self.helpr.get_dens_lsqr_unscale_1p(Pz, lam)
        #print(tdensx)
        "Transition kernel for x -> x2"
        qx = self.helpr.q_CIR_1p(x, Pz, tau, lam)
        qx2 = self.helpr.q_CIR_1p(Pz, x, tau, lam)
        #print(qx.shape)
        alpha = np.minimum(1, (tdensx2 * qx)/(tdensx * qx2))
        #print(alpha)
        unifvec = np.random.uniform(0.0, 1.0)
        if unifvec <= alpha:
            #print(self.helpr.ru(Pz)*self.helpr.rv(Pz))
            return Pz
        
        #print(self.helpr.ru(x)*self.helpr.rv(x))
        
        return x

        


