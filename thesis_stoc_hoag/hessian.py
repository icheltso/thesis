# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:25:19 2023

@author: ichel
"""

import numpy as np
from helpers import Nabla, fmu

class Neumann:
    def __init__(self, A, A_test, y, y_test, ATA, ATy):
        self.A = A
        self.y = y
        self.samples = A.shape[0]
        self.features = A.shape[1]
        self.A_test = A_test
        self.y_test = y_test
        self.samples_t = A_test.shape[0]
        self.features_t = A_test.shape[1]
        self.ATA = ATA    #A^T * A, kept for avoiding expensive computations.
        self.ATy = ATy
        self.grads = Nabla(A, A_test, y, y_test, ATA, ATy)
        
    #Invert directly using sherman-morrison
   # def dir_inverse(self, x, lam):
   #     features = self.features
   #     A = self.A
   #     AAT = self.AAT
   #     v0 = grads.get_nabxC(Dx,x)
   #     I = np.eye(features)
   #     M = (1/(fmu(lam)))*AAT + np.eye(features)
   #     U = (1/(fmu(lam)))*A.T
   #     V =  A
   #     return I - U @ np.linalg.inv(M) @ V
        
    def get_vQ(self, x, lam, out_iter, mu, params):
        s = params["s_outer"]
        grads = self.grads
        # batch sizes for sampling
        Dx=int(np.round(params["C_x_samples"](out_iter,s))+1)
        Dl=params["C_l_samples"]
        eta0=params["vQ_learning_rate"]
        #print("Current outer iteration ", out_iter)
        #print("Current eta ", eta0)
        #print("Current mu ", mu)
        Q = int(np.round(params["Q_for_vQ"](out_iter, s, eta0, mu)) + 1)
        #print("Current Q ",Q)
        #
        eta = eta0
        v0 = grads.get_nabxC(Dx,x)
        #v0 = get_nabxC_dir(x)
        vQ = eta*v0.copy()
        #
        for i in range(1,Q+1):
            B=int(np.round(params["F_xx_samples"](out_iter, s ,eta0, mu, i)) +1)
            vQ = vQ - eta*(grads.get_nabxxF_v(B,vQ,lam) - v0) 
            if params["report"]: 
                print(np.linalg.norm(vQ))
        return vQ
    
    def himplicitEuler(self, x, lam, out_iter, mu, params):
        s = params["s_outer"]
        grads = self.grads
        samples = self.samples
        A = self.A
        Dx=int(np.round(params["C_x_samples"](out_iter,s))+1)
        eta0 = 5*params["vQ_learning_rate"]
        gam = params["vQ_gamma"]
        #B = int(np.round(params["F_xx_samples"](out_iter,s,eta0,mu,out_iter-1)) +1)
        B = 5
        iters = int(np.round(params["Q_for_vQ"](out_iter,s, eta0, mu)) + 1)
        #print(iters)
        v0 = grads.get_nabxC(Dx,x)
        eta = eta0
        #v0 = get_nabxC_dir(x)
        H = eta*v0.copy()
        #Hvals = []
        #Hvals.append(H)
        for i in range(0,iters):
            if (params["vQ_rate_dec"]):
                eta = eta0 * (i+1)**((-1)*gam)
            p = np.random.randint(0,samples,size = B)
            Arow = A[p,:]
            #print(np.size(Arow,0),np.size(Arow,1))
            #Inversion of sampled matrix done by Woodbury formula
            k = samples*eta / (B*(1 + eta*fmu(-100)))
            V = k * Arow
            U = Arow.T
            #IetaAinv = np.eye(features) - U @ np.linalg.inv((np.eye(B) + V @ U)) @ V
            #IetaAinv = (1 / (1 + eta * np.exp(lam))) * IetaAinv
            IetaAinvx = v0 - U @ (np.linalg.inv((np.eye(B) + V @ U)) @ (V @ v0))
            IetaAinvx = (1 / (1 + eta * fmu(-100))) * IetaAinvx
            IetaAinvH = H - U @ (np.linalg.inv((np.eye(B) + V @ U)) @ (V @ H))
            IetaAinvH = (1 / (1 + eta * fmu(-100))) * IetaAinvH
            H = IetaAinvH + eta*IetaAinvx
            #Hvals.append(H)
        
        #return Hvals
        return H