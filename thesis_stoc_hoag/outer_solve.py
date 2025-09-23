# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:07:22 2023

@author: ichel
"""

import numpy as np
import scipy
from scipy import optimize
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator
from helpers import Nabla, fmu
from hessian import Neumann
from inner_solve import InnerSolver


class OuterSolver:
    def __init__(self, A, A_test, y, y_test, ATA, ATy, AAT):
        self.A = A
        self.y = y
        self.samples = A.shape[0]
        self.features = A.shape[1]
        self.A_test = A_test
        self.y_test = y_test
        self.samples_t = A_test.shape[0]
        self.features_t = A_test.shape[1]
        self.ATA = ATA    #A^T * A, kept for avoiding expensive computations.
        self.ATy = ATy    #A^T * y, kept for avoiding expensive computations.
        self.inner = InnerSolver(A, y, ATA, ATy)
        self.grads_o = Nabla(A, A_test, y, y_test, ATA, ATy)
        self.neu = Neumann(A, A_test, y, y_test, ATA, ATy)
            
    def bilvl_gs2(self): #CHECK
        ATA = self.ATA
        ATy = self.ATy
        A_test = self.A_test
        y_test = self.y_test
        features = self.features
    #
        def obj(lam):
            x_min_gs = np.linalg.solve(ATA + fmu(lam)* np.eye(features), ATy)
            Atxy_gs = A_test @ x_min_gs - y_test
            Cval_gs = np.dot(Atxy_gs,Atxy_gs)/2
            return Cval_gs 
    #
        #result=scipy.optimize.minimize_scalar(obj,bounds=(0,1e3),tol=1e-12)
        #result = scipy.optimize.minimize_scalar(obj, method='brent')
        result = scipy.optimize.minimize(obj, x0=0.0, method='BFGS')
        #print(result)
        lam_min=result.x
        x_min_gs = np.linalg.solve(ATA + fmu(lam_min)* np.eye(features), ATy)
        Cvmin=obj(lam_min)
        #
        return fmu(lam_min), Cvmin, x_min_gs
    
    def HOAG_simplified_fix_inn(self, params2):
        ATA = self.ATA
        ATy = self.ATy
        A_test = self.A_test
        y_test = self.y_test
        features = self.features
        inner = self.inner
        grads_o = self.grads_o
        neu = self.neu
        
        params = params2.copy()
        #
        lam_start=params["lam_start"]
        iters=params["max_outer_its"]
        dec_fac = params["dec_fac"]
        #
        #mu=fmu(lam_start)
        #Set the global convexity parameter for the inner problem.
        mu_glob = fmu(-10000)
        #Find the convexity parameter at current iteration - needed to evaluate inner function.
        mu_var = fmu(lam_start)
        #Obtain all the lipschitz terms
        L1 = max(np.linalg.eig(ATA+mu_glob*np.eye(features))[0])
        L2 = max(np.linalg.eig(A_test.T @ A_test)[0])
        L = max(L1,L2)
        L_var = max(np.linalg.eig(ATA+mu_var*np.eye(features))[0])
        #print("L = ", L)
        M = np.linalg.norm(A_test.T @ A_test, 2)
        rho = np.exp(lam_start)
        #get minimum x value
        
        x_min = np.linalg.lstsq(ATA + mu_var*np.eye(features), ATy, rcond=None)[0]
        x_max = np.linalg.lstsq(ATA + fmu(-10000), ATy, rcond=None)[0]
        
        
        #x_min = np.linalg.inv(ATA + mu_var*np.eye(features)) @ ATy
        #Obtain x_max for tau lipschits constant
        #x_max = np.linalg.inv(ATA + fmu(-10000)*np.eye(features)) @ ATy
        tau_mat1 = np.exp(lam_start)*np.eye(features)
        tau_mat2 = np.exp(lam_start)*(x_max-x_min)
        tau_mat = np.transpose(np.block([[tau_mat1],[tau_mat2]]))
        tau = np.linalg.norm(tau_mat,2)
        #Obtain lipschitz constant L_{mathcal{L}} for outer function mathcal{L}(lam)
        L_l = np.real(L + (2*L**2 + tau*M**2)/mu_glob + (rho*L*M + L**3 + tau*M*L)/(mu_glob**2) + (rho*M*L**2)/(mu_glob**3))
        #eta_alg = 1/(3*L)
        #
        eps_base=params["eps_base"]
        eps_decay=params["eps_decay_term"]
        note_C = []
        grad_vals = []
        note_mu = []
        #Q = params["Q_for_vQ"]
        beta0 = (params["learning_rate_outer"]) / L_l
        #print("outer lr: " + str(beta0))
        beta = beta0
        s_out = params["s_outer"]
        
        #In case we want to simulate stocbio, find the maximum inner number of steps in this case
        if params["max_inner_its"] == 'stocbio':
            inner_st_denom = np.log((L+mu_glob)/(L-mu_glob))
            inner_st1 = int(np.log(12+(48*(beta0**2)*(L**2) / mu_glob**2) * (L + (L**2)/mu_glob + M*tau/mu_glob + L*M*rho/mu_glob**2)**2) / (2*inner_st_denom))
            inner_st2 = int(np.log(np.sqrt(beta0) * (L + (L**2)/mu_glob + M*tau/mu_glob + L*M*rho/mu_glob**2)) / inner_st_denom)
            inner_st = max(inner_st1,inner_st2)
            params.update({"max_inner_its":lambda k,s: inner_st})
        #Starting lambda
        lam_alg = lam_start
        x_fin = np.zeros(features)
        #
        for j in range(2,iters):
            #print("Started iteration ", j)
            if params["exact_inner"]:
                #print('Doing exact inner')
                # beware may be ill-posed if ATA singlular and mu small
                #x_fin = np.linalg.solve(ATA + mu_var* np.eye(features), ATy)
                x_fin = np.linalg.lstsq(ATA + mu_var* np.eye(features), ATy, rcond=None)[0]
            elif params["cg"]:
                #print('Doing cg')
                M = LinearOperator((features,features), matvec= lambda x: ATA@x+mu_var*x )
                x_fin, exit_code = linalg.cg(M, ATy, tol = 1/params["max_inner_its"](j-1,s_out), x0 = x_fin)
                #np.size(x_fin,1)
            else:
                if params["eps_dec"]:
                    eps_cur = min(eps_base,1.0 / (j-1)**(0.5+eps_decay))
                else:
                    eps_cur = eps_base
                    #params.update({"max_inner_its":int(j)})
                if params["SGD_sim"]:
                    #print('Doing fast inner sim')
                    #set the mean/variance for noise. offset by 10 to remove instabilities at start
                    sgd_bnd = (1/10)*(4*(L**2)) / (mu_var**2 * params["max_inner_its"](j-1,s_out))
                    mean_coef = (np.abs(1-mu_var/L_var)**params["max_inner_its"](j-1,s_out))
                    #print(mean)
                    x_fin = inner.fake_inner(eps_cur,mu_var, mean_coef, sgd_bnd,x_fin)
                elif params["fast_inner"]:  
                    #print('Doing fast inner')
                    # Lipschitz constant
                    stoch=params["SGD_flag"]
                    x_fin= inner.sgd_solve_inner(x_fin,stoch,eps_cur,mu_glob, mu_var,params["max_inner_its"](j-1,s_out),params["SGD_batch_size"],L_var,np.linalg.norm(x_max-x_min,2))
                    np.size(x_fin,0)
                else:
                    #print('Doing sgd')
                    x_fin,tmp = inner.solve_inner(lam_alg,x_fin,eps_cur,params,j-1)
                
            #
            Atxy = A_test@x_fin - y_test
            Cval = np.dot(Atxy,Atxy)/2
            note_C.append(Cval)
            #
            if params["exact_vQ"]:
                nabxx2f = grads_o.get_nabxxF_dir(lam_alg)
                tmp= np.linalg.lstsq(nabxx2f, grads_o.get_nabxC_dir(x_fin),rcond=None)
                vQ=tmp[0]
            elif (params["neumannEuler"]):
                vQ = neu.get_vQ(x_fin,lam_alg,j,mu_glob,params)
            else:
                vQ = neu.himplicitEuler(x_fin,lam_alg,j,mu_glob,params)
            nablC = grads_o.get_nablC()
            nabxl2F = grads_o.get_nabxl2F(x_fin,lam_alg)
            nabhatL = nablC - np.dot(nabxl2F,vQ)
            grad_vals.append(nabhatL)
            beta = beta0/(j-1)**s_out
            if params["dec_outer"]:
                beta = beta/(j-1)**dec_fac
            lam_alg = lam_alg - beta * nabhatL
            mu_var=fmu(lam_alg)
            print("current mu = ", mu_var)
            L_var = max(np.linalg.eig(ATA+mu_var*np.eye(features))[0])
            #print("current L = ", L_var)
            #print("current inner reciprocal of condition = ", mu_var/L_var)
            #print(mu_var)
            note_mu.append(mu_var)
        print("Finished after ",j," iterations")
        #plt.plot(tmp)
        return fmu(lam_alg), note_C, note_mu, grad_vals