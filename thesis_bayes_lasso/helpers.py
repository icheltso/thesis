# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:51:01 2023

@author: ichel
"""

"Helper functions for finding derivatives and other things"
import os
import jax
import numpy as np
import jax.numpy as jnp
import scipy.integrate as integrate
from scipy.optimize import minimize
#



soft = lambda x,tau: jnp.sign(x)*jnp.maximum(jnp.abs(x)-tau,0)
sigmoid = lambda z: 1 / (1 + jnp.exp(-z))
class Helper:
    def __init__(self, setup):
        current_directory = os.getcwd()
        #file_path = os.path.join(current_directory, "data.npz")
        #file_path = os.path.join(current_directory, "data.pkl")
        #with open(file_path, 'rb') as file:
        #    loaded_dump = dill.load(file)
        #data = np.load(file_path)
        datadict = setup.get_data()
        self.A = setup.A
        self.As = setup.As
        self.lam = datadict["lam"]
        self.n = datadict["n"]
        self.M = datadict["M"]
        self.y = datadict["y"]
        self.y2= datadict["y2"]
        self.sigma = datadict["sigma"]
        self.loss_type = datadict["loss_type"]
        
        #self.A = data["A"]
        #self.n = data["n"]
        #self.M = data["M"]
        #self.y = data["y"]
        #self.y2= data["y2"]
        
    #gradient of moreau approximation potential
    grad_pula = lambda self, x, lam, gamma:  self.dF(x) + (x - soft(x,lam*gamma))/gamma
    grad_pula_1d = lambda self, x, lam, gamma:  self.dF_1d(x) + (x - soft(x,lam*gamma))/gamma
    
        
    def objfun(self,x,Ain,yin):
        if self.loss_type == 'square_loss':
            #print((self.As(self.A(x)-self.y2)).shape)
            return (1/2)*jnp.linalg.norm(Ain(x)-yin,axis=0)**2 + self.lam*jnp.sum(jnp.abs(x),axis=0)
        
        elif self.loss_type == 'log_loss':
            z = yin*Ain(x)
            outlog = 0
            for i in range(z.shape[0]):
                outlog += jnp.log(1 + jnp.exp(-z[i]))
            return outlog + self.lam*jnp.sum(jnp.abs(x),axis=0)
            
    
    def dF(self,x):
        if self.loss_type == 'square_loss':
            #print((self.As(self.A(x)-self.y2)).shape)
            return self.As(self.A(x)-self.y2)
        elif self.loss_type == 'log_loss':
            #return self.As(sigmoid(self.A(x)) - self.y2)
            #return self.As((sigmoid(self.y2 * self.A(x)) - 1) * self.y2)
            return self.As(-sigmoid(-self.y2 * self.A(x)) * self.y2)
    
    def dF_1d(self,x):
        if self.loss_type == 'square_loss':
            return self.As(self.A(x)-self.y)
        elif self.loss_type == 'log_loss':
            #return self.As(sigmoid(self.A(x)) - self.y)
            #return self.As((sigmoid(self.y * self.A(x)) - 1) * self.y)
            return self.As(-sigmoid(-self.y * self.A(x)) * self.y)
    
    
    #dF = lambda self, x: self.As(self.A(x)-self.y2)
    #dF_1d = lambda self, x: self.As(self.A(x)-self.y)
    
    
    
    #Forward-Backward Gradient for EULA
    def gradFB(self, x, lam, gamma):
        z = x - soft( x - gamma*self.dF(x), gamma*lam)
        return (z - gamma*self.As(self.A(z)))/gamma
    
    #u,v separation
    #ru = lambda self, x: x[:self.n,:]
    #rv = lambda self, x: x[self.n:,:]
    #ru = lambda self, x: x[:self.n]
    #rv = lambda self, x: x[self.n:]
    
    ru = lambda self, x: x[:len(x[:,0])//2]
    rv = lambda self, x: x[len(x[:,0])//2:]
    
    u_sq = lambda self,y,z: (y*y + z*z)**(1/2)
    
    #hadamard gradient
    grad = lambda self,x,g: jnp.concatenate((self.rv(x)*g , self.ru(x)*g))
    grad_1d = lambda self,x,g: np.concatenate((self.rv(x)*g , self.ru(x)*g),0)
    Grd = lambda self,x: self.grad( x, self.dF(self.ru(x)*self.rv(x)) )
    Grd_1d = lambda self,x: self.grad_1d( x, self.dF_1d(self.ru(x)*self.rv(x)) )
    
    def Grd_3(self, x):
        
        #z1 = x[:len(x[:,0])//3,:]
        #z2 = x[len(x[:,0])//3:2*len(x[:,0])//3,:]
        #v = x[2*len(x[:,0])//3:,:]
        
        z1 = x[:self.n,:]
        z2 = x[self.n:2*self.n,:]
        v = x[2*self.n:,:]
        
        u_sq_v = self.u_sq(z1,z2)
        df_save = self.dF(u_sq_v * v)
        
        Grd_3_1 = (-z1/u_sq_v) * v * df_save
        Grd_3_2 = (-z2/u_sq_v) * v * df_save
        Grd_3_3 = -u_sq_v * df_save
        return jnp.concatenate((Grd_3_1,Grd_3_2,Grd_3_3), axis=0)
    
    #
    
    def q_CIR(self, x, x_2, tau, lam):
        "Transition density for one step of sqrt-CIR. Needed for metropolis-adjusted algorithm."
        beta = 2 / self.sigma**2
        b = (1+lam*tau)
        #f_0 = lambda z1,z2: jnp.exp((1/4*tau)*jnp.linalg.norm(z1 - (z2 - tau*self.Grd(z2)),2)**2)
        f_0 = lambda z1,z2: jnp.exp(beta*(1/(4*tau))*(z1 - (z2 - tau*self.Grd(z2)))**2)
        f1u = lambda z1,z2: self.ru(f_0(b*z1 - tau/(beta*z1),z2)) * (b + tau/(beta*self.ru(z1)**2))
        f1v = lambda z1,z2: self.rv(f_0(b*z1,z2))
        
        return jnp.prod(f1u(x_2,x) * f1v(x_2,x), 0)
    
    def q_CIR_1p(self, x, x_2, tau, lam):
        "Transition density for one step of sqrt-CIR. Needed for metropolis-adjusted algorithm."
        "Single particle version"
        beta = 2 / self.sigma**2
        b = (1+lam*tau)
        f_0 = lambda z1,z2: np.exp(beta*(1/(4*tau))*(z1 - (z2 - tau*self.Grd_1d(z2)))**2)
        f1u = lambda z1,z2: self.ru(f_0(b*z1 - tau/(beta*z1),z2)) * (b + tau/(beta*self.ru(z1)**2))
        f1v = lambda z1,z2: self.rv(f_0(b*z1,z2))
        
        #print(f_0(b*x_2 - tau/x_2,x))
        #print(np.prod(f1u(x_2,x) * f1v(x_2,x)))
        
        return np.prod(f1u(x_2,x) * f1v(x_2,x))
    
    def q_CIR_old(self, x, x_2, tau, lam):
        "Transition density for one step of sqrt-CIR. Needed for metropolis-adjusted algorithm."
        u = self.ru(x)
        v = self.rv(x)
        u2 = self.ru(x_2)
        v2 = self.rv(x_2)
        b = 2*(1+lam*tau)
        ua = u - tau*v*self.dF(u*v)
        "Transition kernel from u to (ua + xi)/b"
        base_inter = lambda z: b/jnp.sqrt(4*jnp.pi*tau) * jnp.exp(-(b**2 * (z - ua/b)**2) / (4*tau))
        ginv = lambda z: (z**2 - (2*tau/b))/(2*z)
        ginv_grad = lambda z: (z**2 + (2*tau/b))/(2*z**2)
        "Use change of variables formula"
        q_u_cir = base_inter(ginv(u2)) * jnp.abs(ginv_grad(u2))
        q_v_cir = (b/2)/jnp.sqrt(4*jnp.pi*tau) * jnp.exp(-((b/2)**2 * ( v2 - 2*v/b + 2*tau*u*self.dF(u*v)/b))**2 / (4*tau))
        return jnp.prod(q_u_cir,0) * jnp.prod(q_v_cir,0)
        
        
    def get_dens_lsqr(self, xt, lam):
        """Generate densities corresponding to lam_R|z|_1 + G(z)"""
        """Only works in 1-dim case"""
        Anp = self.A
        y2np = np.array(self.y2)
        xs = np.array(xt)
        print(xs.shape)
        
        lr = np.array(lam)
        
        ynp = np.array(self.y)
        
        #np.asarray(jax.device_get(Anp(x)))
        
        sigma=np.sqrt(2)
        beta = 2/sigma**2
        sz = np.size(xs)
        denszs = np.zeros(sz)
        Z_out, Z_err = integrate.quad(lambda x: np.exp(-beta * (lr*np.abs(x) + 1/2*np.linalg.norm(np.asarray(jax.device_get(Anp(jnp.array([x]))))-ynp,2)**2)), -np.inf, np.inf)
        for i in range(0,sz):
            denszs[i] = (1/Z_out)*np.exp(-beta*(lr*np.abs(xs[i]) + 1/2*np.linalg.norm(np.asarray(jax.device_get(Anp(jnp.array([xs[i]]))))-ynp,2)**2))
        return jnp.array(denszs)
    
    def get_dens_lsqr_unscale(self, xt, lam):
        """Generate target densitiy: exp(-beta*G(u,v)), with G(u,v) = lam * (|u|^2 + |v|^2) + 1/2|Ax-b|^2. Doesn't scale the density so that it integrates to 1. Needed for Metropolis Hastings."""
        xtarg = self.ru(xt) * self.rv(xt)
        tn,tM = xtarg.shape
        sigma=np.sqrt(2)
        beta = 2/sigma**2
        #print(tM)
        #denszs = jnp.exp(-beta*(lam*jnp.linalg.norm(xtarg,1)**2 + 1/2*jnp.linalg.norm(self.A@xtarg-self.y2,2)**2))
        #denszs = jnp.prod(self.ru(xt),0) * jnp.exp(-beta*((lam/2)*jnp.linalg.norm(xt,2)**2 + 1/2*jnp.linalg.norm(self.A@xtarg-self.y2,2)**2))
        denszs = jnp.prod(self.ru(xt),0) * jnp.exp(-beta*((lam/2)*jnp.sum(xt**2,0) + 1/2*jnp.sum((self.A(xtarg)-self.y2)**2,0)))
        #print(denszs.shape)
        return denszs
    
    def get_dens_lsqr_unscale_1p(self, xt, lam):
        """Generate densities corresponding to lam_R|z|_1 + G(z). Doesn't scale the density so that it integrates to 1. Needed for Metropolis Hastings."""
        xtarg = self.ru(xt) * self.rv(xt)
        #sigma=np.sqrt(2)
        #beta = 2/sigma**2
        beta = 1
        
        #denszs = np.exp(-beta*(lam*np.linalg.norm(xtarg,1)**2 + 1/2*np.linalg.norm(self.A(xtarg)-self.y.reshape(-1,1),2)**2))
        
        denszs = np.prod(self.ru(xt),0) * np.exp(-beta*((lam/2)*np.sum(xt**2,0) + 1/2*np.sum((self.A(xtarg)-self.y)**2)))
        #denszs = np.exp(-beta*((lam/2)*jnp.sum(xt**2,0) + 1/2*np.sum((self.A(xtarg)-self.y.reshape(-1,1))**2,0)))
        
        return denszs
        
    
    def solve_lasso_lsqr(self, lam):
        #Anp = np.array(self.A)
        ynp = np.array(self.y)
        #lasso = lambda x: lam*np.sum(np.abs(x)) + (1/2)*np.linalg.norm(Anp@x-ynp,2)**2
        lasso = lambda x: lam*np.sum(np.abs(x)) + (1/2)*np.linalg.norm(self.A(x)-ynp,2)**2
        #lasso = lambda x: lam*np.linalg.norm(x,1) + (1/2)*np.linalg.norm(Anp@x-ynp,2)**2
        optim = minimize(lasso, np.zeros((self.n,)), method='L-BFGS-B')
        return optim.x
        

