# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:01:17 2023

@author: ichel
"""
import pickle as pkl
import numpy as np
#from inner_solve import InnerSolver
from outer_solve_old import OuterSolver
#from setup import A, A_test, y, y_test, ATA, ATy, AAT
#
import matplotlib.pyplot as plt


##############################
###TESTS
##############################
[A, A_test, y, y_test, ATA, ATy, AAT] = pkl.load(open('file.pkl', 'rb'))


solver = OuterSolver(A, A_test, y, y_test, ATA, ATy, AAT)

#Obtain exact solution
minmu, minC, x = solver.bilvl_gs2()
print("Optimal hyperparameter mu=",minmu, "\nOptimal objective function L=",minC)
print("Norm of x is ",np.linalg.norm(x))
print("Norm of Ax-y is ",np.linalg.norm(A@x-y))

print("Norm of A_t x-y_t is ",np.linalg.norm(A_test @x-y_test))
print("Norm of A is ",np.linalg.norm(A))



#Obtain solution using pseudo-sgd for the inner problem.

#number of iterations of outer loop
max_its = int(20000)

#Parameters for STOC-HOAG
params_hoag={"lam_start":2,
        "max_outer_its":max_its,
        "learning_rate_outer":400,  #Base outer stepsize
        "s_outer":0, #Choose between 0 and 0.5
        "dec_outer": False,
        "dec_fac": 0.5,
        # inner flags
        "exact_inner":False,
        "cg": False,                #Solve the inner problem using conjugate gradient.
        "SGD_flag":True,            #If not exactly computing the inner problem, True for SGD, False for GD
        "record_sgd": True,
        "SGD_sim": False,
        "sim_noise_mean": lambda k: 0.05/k, #noise for simulated SGD solution error. set 1/k for strongly convex, 1/sqrt(k) otherwise.
        #"sim_noise_mean": lambda k: 0,
        #"max_inner_its": lambda k,s: int(np.round(k**(1-s))+1),
        "max_inner_its": lambda k,s: int(np.sqrt(np.round(k**(1-s)))+1),
        #"max_inner_its": lambda k,s: int(np.log(np.round(k**(1-s)))+1),
        "fast_inner":True,
        "eps_base":10000,# tolerance for inner problem
        "eps_dec": False, #epsilon decaying T/F?
        "eps_decay_term":0.5, #If epsilon decaying then eps_t = eps_0/t^(0.5+edc)
        "sgd_lr": lambda k,mu: (1/mu)*  (1.0/(k+1)), # sgd learning rate for iterate k, mu-st.conv.
        "SGD_batch_size":10,
        # vQ params
        "exact_vQ": False,
        "neumannEuler": True,
        "vQ_learning_rate":0.125,# (aka, eta)
        "vQ_rate_dec": False, #whether learning rate for implicit euler is a decaying, square-summable sequence.
                             #by default, set eta = eta0 * n^(-gamma) for gamma in (0.5,1]
        "vQ_gamma": 0.5, #We set gamma for the v_Q learning rate.
        #"Q_for_vQ": lambda k, eta, mu: 30,  #Constant value for Q
        "Q_for_vQ": lambda k, s, eta, mu: (np.log(k**(1-s)) / (2*(-1)*np.log(1-eta*mu))),  #Increasing Q
        "C_x_samples": lambda k, s: k**(1-2*s),
        "C_l_samples":5,
        "F_xx_samples": lambda k,s,eta,mu,j: (k**(1-2*s))*(1-eta*mu)**(j-1),
        #"F_xx_samples": lambda k: ,
        # other
        "report":False,
         }

hoag_lis_dict = {"max_inner_its": lambda k,s: int(np.log(np.round(k**(1-s)))+1)}
params_hoag_lis = params_hoag.copy()
params_hoag_lis.update(hoag_lis_dict)

gd_dict = {("SGD_flag",False)}
params_gd = params_hoag.copy()
params_gd.update(gd_dict)

pos_s_dict = [("max_inner_its", lambda k,s: int(np.round(k**(1-s))+1)),("s_outer",0.5),("dec_outer", False)]
params_hoag_pos_s = params_hoag.copy()
params_hoag_pos_s.update(pos_s_dict)


isgd_dict = {"neumannEuler": False}
params_hoag_isgd = params_hoag.copy()
params_hoag_isgd.update(isgd_dict)

cg_dict = [("cg", True),("max_inner_its", lambda k,s: int(np.round(k**(1-s))+1))]
params_hoag_cg = params_hoag.copy()
params_hoag_cg.update(cg_dict)

#Parameters for stocbio
stocbio_dict = [("sgd_lr",0.01),("C_x_samples", lambda k, s:5),("max_inner_its", 'stocbio')]
params_stocbio = params_hoag.copy()
params_stocbio.update(stocbio_dict)

#params_stocbio={"lam_start":2,
#        "max_outer_its":max_its,
#        "learning_rate_outer":50,  #Base outer stepsize
#        "s_outer":0, #Choose between 0 and 0.5
#        "dec_outer": False,
#        # inner flags
#        "exact_inner":False,
#        "cg": False,                #Solve the inner problem using conjugate gradient.
#        "SGD_flag":True,            #If not exactly computing the inner problem, True for SGD, False for GD
#        "record_sgd": True,
#        "SGD_sim": False,
#        "sim_noise_mean": lambda k: 0.05/k, #noise for simulated SGD solution error. set 1/k for strongly convex, 1/sqrt(k) otherwise.
#        #"sim_noise_mean": lambda k: 0,
#        "max_inner_its": lambda k,s: int(np.round(k**(1-s))+1),
#        "fast_inner":True,
#        "eps_base":10000,# tolerance for inner problem
#        "eps_dec": False, #epsilon decaying T/F?
#        "eps_decay_term":0.5, #If epsilon decaying then eps_t = eps_0/t^(0.5+edc)
#        #"sgd_lr": lambda k,mu: (1/mu)*  (1.0/(k+1)), # sgd learning rate for iterate k, mu-st.conv.
#        "sgd_lr": lambda k,mu: 0.01, # sgd learning rate for stocbio.
#        "SGD_batch_size":5,
#        # vQ params
#        "exact_vQ": False,
#        "neumannEuler": True,
#        "vQ_learning_rate":0.125,# (aka, eta)
#        "vQ_rate_dec": False, #whether learning rate for implicit euler is a decaying, square-summable sequence.
                             #by default, set eta = eta0 * n^(-gamma) for gamma in (0.5,1]
#        "vQ_gamma": 0.5, #We set gamma for the v_Q learning rate.
#        #"Q_for_vQ": lambda k, eta, mu: 30,  #Constant value for Q
#        "Q_for_vQ": lambda k, s, eta, mu: (np.log(k**(1-s)) / (2*(-1)*np.log(1-eta*mu))),  #Increasing Q
#        "C_x_samples": lambda k, s: 5,
#        "C_l_samples":5,
#        "F_xx_samples": lambda k,s,eta,mu,j: (k**(1-2*s))*(1-eta*mu)**(j-1),
#        #"F_xx_samples": lambda k: ,
#        # other
#        "report":False,
#         }

runs = 30
opt_mu_runs = np.zeros(7)
gradient_runs = np.zeros((7,max_its-2))
feval_runs = np.zeros((7,max_its-2))
mus_runs = np.zeros((7,max_its-2))

for j in range(runs):
    print("Started run " + str(j) + " of " + str(runs))
    print("\nBegin Stoc-Hoag (SIS) with s=0")
    mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_hoag)
    print("Size of gradient of objective", gradient[-1])
    print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    opt_mu_runs[0] = opt_mu_runs[0] + mu_from_alg
    gradient_runs[0,:] = gradient_runs[0,:] + gradient
    feval_runs[0,:] = feval_runs[0,:] + feval_alg
    mus_runs[0,:] = mus_runs[0,:] + the_mus
    
    #print("\nBegin Stoc-Hoag (LIS) with s=0")
    #mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_hoag_lis)
    #print("Size of gradient of objective", gradient)
    #print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    #print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    #opt_mu_runs[1] = opt_mu_runs[1] + mu_from_alg
    #gradient_runs[1,:] = gradient_runs[1,:] + gradient
    #feval_runs[1,:] = feval_runs[1,:] + feval_alg
    #mus_runs[1,:] = mus_runs[1,:] + the_mus
    
    #print("\nBegin Stoc-Hoag with s=0.5")
    #mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_hoag_pos_s)
    #print("Size of gradient of objective", gradient[-1])
    #print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    #print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    #opt_mu_runs[2] = opt_mu_runs[2] + mu_from_alg
    #gradient_runs[2,:] = gradient_runs[2,:] + gradient
    #feval_runs[2,:] = feval_runs[2,:] + feval_alg
    #mus_runs[2,:] = mus_runs[2,:] + the_mus
    
    print("\nBegin Stoc-Hoag (s=0) with CG for inner problem")
    mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_hoag_cg)
    print("Size of gradient of objective", gradient[-1])
    print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    opt_mu_runs[3] = opt_mu_runs[3] + mu_from_alg
    gradient_runs[3,:] = gradient_runs[3,:] + gradient
    feval_runs[3,:] = feval_runs[3,:] + feval_alg
    mus_runs[3,:] = mus_runs[3,:] + the_mus
    
    print("\nBegin Stocbio")
    mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_stocbio)
    print("Size of gradient of objective", gradient[-1])
    print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    opt_mu_runs[4] = opt_mu_runs[4] + mu_from_alg
    gradient_runs[4,:] = gradient_runs[4,:] + gradient
    feval_runs[4,:] = feval_runs[4,:] + feval_alg
    mus_runs[4,:] = mus_runs[4,:] + the_mus
    
    #print("\nBegin Stoc-Hoag (s=0) with GD for inner problem")
    #mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_gd)
    #print("Size of gradient of objective", gradient[-1])
    #print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    #print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    #opt_mu_runs[5] = opt_mu_runs[5] + mu_from_alg
    #gradient_runs[5,:] = gradient_runs[5,:] + gradient
    #feval_runs[5,:] = feval_runs[5,:] + feval_alg
    #mus_runs[5,:] = mus_runs[5,:] + the_mus
    
    print("Begin Stoc-Hoag (s=0) with ISGD for Hessian")
    mu_from_alg, feval_alg, the_mus, gradient = solver.HOAG_simplified_fix_inn(params_hoag_isgd)
    print("Size of gradient of objective", gradient)
    print("Optimal mu (HOAG)=",mu_from_alg, ", reference=",minmu)
    print("Objective (HOAG)=",feval_alg[-1], ", reference=",minC)
    opt_mu_runs[6] = opt_mu_runs[6] + mu_from_alg
    gradient_runs[6] = gradient_runs[6] + gradient
    feval_runs[6,:] = feval_runs[6,:] + feval_alg
    mus_runs[6,:] = mus_runs[6,:] + the_mus
    
opt_mu_runs = opt_mu_runs / runs
gradient_runs = gradient_runs / runs
feval_runs = feval_runs / runs
mus_runs = mus_runs / runs

not_transient = max_its
    
    
print("Optimal mu for Stoc-Hoag (SIS) with s=0 =",opt_mu_runs[0])
#print("Optimal mu for Stoc-Hoag (LIS) with s=0 =",opt_mu_runs[1])
#print("Optimal mu for Stoc-Hoag with s=0.5 =",opt_mu_runs[2])
print("Optimal mu for Stoc-Hoag with s=0 and CG inner problem =",opt_mu_runs[3])
print("Optimal mu for Stoc-Hoag with s=0 and ISGD hessian =",opt_mu_runs[6])
print("Optimal mu for Stocbio =",opt_mu_runs[4])
#print("Optimal mu for Stoc-Hoag with s=0 and GD inner problem =",opt_mu_runs[5])
print("Objective for Stoc-Hoag (SIS) with s=0 =",feval_runs[0,-1])
#print("Objective for Stoc-Hoag (LIS) with s=0 =",feval_runs[1,-1])
#print("Objective for Stoc-Hoag with s=0.5 =",feval_runs[2,-1])
print("Objective for Stoc-Hoag with s=0 and CG inner problem =",feval_runs[3,-1])
print("Objective for Stoc-Hoag with s=0 and ISGD hessian=",feval_runs[6,-1])
print("Objective for Stocbio=",feval_runs[4,-1])
#print("Objective for Stoc-Hoag with s=0 and GD inner problem =",feval_runs[5,-1])
lspace = np.array(range(1,max_its-1))
notes_curve = np.log(lspace) / np.sqrt(lspace)
sqrt_curve = 1 / np.sqrt(lspace)
thry_curve = np.log(lspace) / lspace
linvspace = 1/(lspace)
linvspace2 = 1/(lspace**2)
#tofit_y = np.log(np.abs(feval_runs[-not_transient:] - minC) / np.abs(minC))
#tofit_x = np.log(lspace[-not_transient:])
#z = np.polyfit(tofit_x, tofit_y, 1)
#polz = np.poly1d(z)
#raised_to_pow = np.exp(z[0]*np.log(lspace) + z[1])
plt.figure(1)
plt.loglog(np.abs(feval_runs[0,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (SIS) (s=0)')
#plt.loglog(np.abs(feval_runs[1,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (LIS) (s=0)')
#plt.loglog(np.abs(feval_runs[2,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0.5)')
plt.loglog(np.abs(feval_runs[4,:] - minC) / np.abs(minC), label = 'Stocbio')
plt.loglog(np.abs(feval_runs[3,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0, CG)')
#plt.loglog(np.abs(feval_runs[5,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0, GD)')
plt.loglog(np.abs(feval_runs[6,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0, ISGD Hess)')
#plt.loglog(lspace,raised_to_pow, label = 'Fitted Line')

#plt.plot(tofit_x,polz(tofit_x), label = 'Fitted Log-Line')
#plt.loglog(polz(lspace), label = 'Fitted Line')

plt.loglog(linvspace, linestyle='--', label = '1/t')
#plt.loglog(sqrt_curve, linestyle='--', label = '1/sqrt(t)')
#plt.loglog(notes_curve, linestyle='--', label = 'log(t)/sqrt(t)')
plt.loglog(thry_curve, linestyle='--', label = 'log(t)/t')
plt.legend()
tmp=plt.xlabel("Iterations")
tmp=plt.ylabel("$|L(\lambda) - L_\min| / |L_\min|$")
plt.savefig("a_run_of_STHOAG.jpg", bbox_inches ="tight")
plt.savefig('a_run_of_STHOAG.eps', format='eps', bbox_inches ="tight")

#plt.semilogy(np.abs(feval_runs[0,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (SIS) (s=0)')
#plt.semilogy(np.abs(feval_runs[1,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (LIS) (s=0)')
#plt.semilogy(np.abs(feval_runs[2,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0.5)')
#plt.semilogy(np.abs(feval_runs[4,:] - minC) / np.abs(minC), label = 'Stocbio')
#plt.semilogy(np.abs(feval_runs[3,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0, CG)')
#plt.semilogy(np.abs(feval_runs[5,:] - minC) / np.abs(minC), label = 'Stoc-HOAG (s=0, GD)')
#plt.loglog(lspace,raised_to_pow, label = 'Fitted Line')

#plt.plot(tofit_x,polz(tofit_x), label = 'Fitted Log-Line')
#plt.loglog(polz(lspace), label = 'Fitted Line')

#plt.semilogy(linvspace, linestyle='--', label = '1/t')
#plt.semilogy(sqrt_curve, linestyle='--', label = '1/sqrt(t)')
#plt.semilogy(notes_curve, linestyle='--', label = 'log(t)/sqrt(t)')
#plt.semilogy(thry_curve, linestyle='--', label = 'log(t)/t')
#plt.legend()
#tmp=plt.xlabel("Iterations")
#tmp=plt.ylabel("$|L(\lambda) - L_\min| / |L_\min|$")
#plt.savefig("a_run_of_STHOAG_semilogy.jpg", bbox_inches ="tight")
#plt.savefig('a_run_of_STHOAG_lin.eps', format='eps', bbox_inches ="tight")
print(mus_runs)
#print("Rate of convergence (via fitted line)", z[0])

#Plot iterate differences
plt.figure(2)
plt.loglog(np.abs(mus_runs[0] - minmu) / np.abs(minmu), label = 'Stoc-HOAG')
#plt.loglog(np.abs(mus_runs[1] - minmu) / np.abs(minmu), label = 'Stoc-HOAG (LIS) (s=0)')
#plt.loglog(np.abs(mus_runs[2] - minmu) / np.abs(minmu), label = 'Stoc-HOAG (s=0.5)')
plt.loglog(np.abs(mus_runs[4] - minmu) / np.abs(minmu), label = 'Stocbio')
plt.loglog(np.abs(mus_runs[3] - minmu) / np.abs(minmu), label = 'Stoc-HOAG (CG)')
#plt.loglog(np.abs(mus_runs[5] - minmu) / np.abs(minmu), label = 'Stoc-HOAG (s=0, GD)')
plt.loglog(np.abs(mus_runs[6] - minmu) / np.abs(minmu), label = 'Stoc-HOAG (ISGD)')
plt.loglog(linvspace, linestyle='--', label = '1/t')
#plt.loglog(sqrt_curve, linestyle='--', label = '1/sqrt(t)')
#plt.loglog(notes_curve, linestyle='--', label = 'log(t)/sqrt(t)')
plt.loglog(thry_curve, linestyle='--', label = 'log(t)/t')
plt.legend()
tmp=plt.xlabel("Iterations")
tmp=plt.ylabel("$|\lambda - \lambda_\min| / |\lambda_\min|$")
plt.savefig("a_run_of_STHOAG_its.jpg", bbox_inches ="tight")
plt.savefig('a_run_of_STHOAG_its.eps', format='eps', bbox_inches ="tight")

plt.figure(3)
plt.loglog(np.abs(gradient_runs[0]), label = 'Stoc-HOAG')
#plt.loglog(gradient_runs[1], label = 'Stoc-HOAG (LIS) (s=0)')
#plt.loglog(np.abs(gradient_runs[2]), label = 'Stoc-HOAG (s=0.5)',alpha = 0.4)
plt.loglog(np.abs(gradient_runs[4]), label = 'Stocbio', alpha = 0.4)
plt.loglog(gradient_runs[3], label = 'Stoc-HOAG (CG)')
#plt.loglog(np.abs(gradient_runs[5]), label = 'Stoc-HOAG (s=0, GD)', alpha = 0.5)
plt.loglog(np.abs(gradient_runs[6]), label = 'Stoc-HOAG (ISGD)', alpha = 0.5)
plt.loglog(linvspace, linestyle='--', label = '1/t')
#plt.loglog(sqrt_curve, linestyle='--', label = '1/sqrt(t)')
#plt.loglog(notes_curve, linestyle='--', label = 'log(t)/sqrt(t)')
plt.loglog(thry_curve, linestyle='--', label = 'log(t)/t')
#plt.loglog(linvspace2, linestyle='--', label = '1/t^2')
plt.legend()
tmp=plt.xlabel("Iterations")
tmp=plt.ylabel(''r'$\| \nabla L(\lambda) \|$')
plt.savefig("a_run_of_STHOAG_grad.jpg", bbox_inches ="tight")
plt.savefig('a_run_of_STHOAG_grad.eps', format='eps', bbox_inches ="tight")


z=np.zeros((2,6))

for i in range(6):
    tofit_y = np.log(np.abs(feval_runs[i,-not_transient:] - minC) / np.abs(minC))
    tofit_x = np.log(lspace[-not_transient:])
    z[:,i] = np.polyfit(tofit_x, tofit_y, 1)