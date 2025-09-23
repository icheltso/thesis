# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:49:53 2023

@author: ichel
"""
from numba import jit
import numpy as np
from helpers import fmu

class InnerSolver:
    def __init__(self, A, y, ATA, ATy):
        self.A = A
        self.samples = A.shape[0]
        self.features = A.shape[1]
        self.y = y
        self.ATA = ATA    #A^T * A, kept for avoiding expensive computations.
        self.ATy = ATy    #A^T * y, kept for avoiding expensive computations.


    def fake_inner(self, eps, mu, mean_coef, sgd_bnd, x_warm):
        features = self.features
        ATA = self.ATA
        ATy = self.ATy
        x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        xdiffs = np.linalg.norm(x_warm - x_opt,2)
        mean = np.sqrt(mean_coef * xdiffs**2)
        sgd_bnd2 = sgd_bnd * xdiffs**2
        #print("Mean = ", mean)
        #var = np.abs(sgd_bnd - mean**2)
        var = sgd_bnd
        #print("Variance = ", var)
        epsilon = np.random.normal(mean, np.sqrt(var), features)
        return x_opt + epsilon
    
    #@jit
    def sgd_solve_inner(self, x0,stoch,eps,mu_glob,mu,tmax,B,L,x_maxmin):
        A = self.A
        y = self.y
        features = self.features
        samples = self.samples
        ATA = self.ATA
        ATy = self.ATy
        # initialise
        x_fin = x0
        # exact solution
        #x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        #eps = (4*x_maxmin*L**2*np.linalg.norm(x0,2)**2) / (tmax * mu**2)
        #counter = 0
        for t in range(0,tmax):
            # set learning rate 
            alpha = 1.0/(mu*(t+1))
            # choose sample
            if (stoch == True):
                p = np.random.randint(0,samples,size = B)
                As = A[p,:]
                ys = y[p]
                # approximate gradient
                gradF = np.real((samples/B)*As.T @ (As @ x_fin - ys) + (mu)*x_fin)
            else:
                gradF = ATA @ x_fin - ATy + (mu)*x_fin
            # update
            x_fin = x_fin - alpha * gradF
            # 
            #if (np.linalg.norm(x_fin - x_opt,2)**2 < eps):
            #    break
        #error=np.linalg.norm(x_fin - x_opt,2)**2
        #if (error>eps):
        #    print("Inner iteration fails to converge within "+ str(tmax) + " iterations")
        #    print("eps = ", eps)
        #    print("error = ",error)
        #    print("stepsize, alpha")
        #   assert(False)
        #else:
        #print("Passed inner iteration ", t, "iterations; error=", error, ", tolerance= ", eps, "alpha=",alpha)
        return x_fin
###################################################


    def solve_inner(self,lam,x0,eps,params,k):
        A = self.A
        y = self.y
        features = self.features
        samples = self.samples
        ATA = self.ATA
        ATy = self.ATy
        s = params["s_outer"]
    
        tmax=params["max_inner_its"](k,s)
        print("max inner its " + str(tmax))
        stoch=params["SGD_flag"] # GD or SGD
        B=params["SGD_batch_size"]
        assert(B>0)
        note_fn = []
        # Lipschit constant
        mu_L=float(fmu(-10000))
        mu = float(fmu(lam))
        L = max(np.abs(np.linalg.eig(ATA+mu_L*np.eye(features))[0]))

        # initialise
        x_fin = x0
        # exact solution
        x_opt = np.linalg.solve(ATA + mu * np.eye(features), ATy)
        #
        counter = 0
        for t in range(0,tmax):
            counter += 1
            # set learning rate
            alpha = np.real(params["sgd_lr"](counter,mu))
            assert(alpha>0)
            #
            if (stoch == True):
                # choose sample
                p = np.random.randint(0,samples,size = B)
                As = A[p,:]
                ys = y[p]
                # approximate gradient
                gradF = np.real((samples/B)*As.T @ (As @ x_fin - ys) + (mu)*x_fin)
            else:
                gradF = ATA @ x_fin - ATy + (mu)*x_fin
            #
            x_fin = x_fin - alpha * gradF
            #
            if  params["record_sgd"]:
                #note_xes.append(x_fin)
                Axy = A@x_fin - y
                fnval = np.dot(Axy,Axy)/2 + (mu/2)*np.dot(x_fin,x_fin)
                note_fn.append(fnval)
                #print("Inner SGD fn value: " + str(fnval))
            if (np.linalg.norm(x_fin - x_opt,2) < eps):
                break
        error=np.linalg.norm(x_fin - x_opt,2)
        print(error)
        if (error>eps):
            print("Inner iteration fails to converge within "+ str(tmax) + " iterations")
            print("eps = ", eps)
            print("error = ",error)
            assert(False)
        if params["report"]:
            print("Inner error=",error, "x=",x_fin)
    
        #print("Finished inner problem in " + str(counter) + " iterations")

        return x_fin, note_fn
    
    
