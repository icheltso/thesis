# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:09:50 2024

@author: ichel
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
#from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_breast_cancer
import pandas as pd

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft, ifft
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
#import arviz as az
from jax.numpy.fft import fft, ifft
from skimage.util import random_noise
from runner import Runner
from helpers import Helper
from setup import Setup
from util import GaussianFilter, getWaveletTransforms_jax, legend_without_duplicate_labels
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from statsmodels.tsa.stattools import acf
import arviz as az
import numpy.random as rnd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


save_xtra = os.path.join("SIMULATION","BINARY","SYNTH","BELAM")
os.makedirs(save_xtra, exist_ok=True)


"Package relevant parameters into a single .npz file, which will be accessed by all other modules."
def setup_data(A,n,lam,y,M,gamma,beta,key):
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
    print(y2.shape)
    
    Af = lambda x: A @ x
    As = lambda x: A.T @ x
    
    'sde sigma'
    sigma = jnp.sqrt(2/beta)
    
    'Parameter R - for constrained uv method'
    R = 1000
    
    #todump = {A, As, n, lam, np.array(y), np.array(y2), np.array(initx), np.array(initv), M, gamma}
    
    'For loss_type, choose between square_loss and log_loss'
    #loss_type = "square_loss"
    loss_type = "log_loss"
    
    setup_data = Setup(sigma, loss_type, Af, As, n, lam, y, y2, initx, initv, initz, M, gamma, R)
    
    #np.savez(file_path, A=A, As=As, n=n, lam=lam, x0=x0, y=y, y2=y2, initx=initx, initv=initv, M=M, gamma=gamma)
    #with open(file_path, 'wb') as file:
    #    dill.dump(todump, file)
        
    return setup_data


def rFISTA(proxF, dG, gamma, xinit,niter,mfunc):
    tol = 1e-16

    x = xinit
    z = x
    t=1
    fval = []
    adj_diff = []
    for k in range(niter):
        xkm = x
        ykm = z
        
        x =  proxF( z - gamma*dG(z), gamma )
        tnew = (1+np.sqrt(1+4*t**2))/2

        z = x + (t-1)/(tnew)*(x-xkm)
        t = tnew
        if np.sum((ykm-x)*(x-xkm))>0:
            z=x;
        fval.append(mfunc(x))
        adj_diff.append(np.linalg.norm(x - xkm))

        if np.linalg.norm(xkm-x)>tol and k == niter-1:
            print(f"rFISTA didn't converge in max iters")
        
        if np.linalg.norm(xkm-x)<tol:
        #    print(f"Final consecutive difference for restarted FISTA: {np.linalg.norm(xkm-x)} at iteration {k}")
            break
        
    #print(f"Final consecutive difference for restarted FISTA: {np.linalg.norm(xkm-x)}")
    return x, fval, adj_diff

def ISTA(proxF, dG, gamma, xinit,niter,mfunc):
    x = xinit
    
    fval = []
    adj_diff = []
    for k in range(niter):
        x_old = x
        x =  proxF( x - gamma*dG(x), gamma )
        fval.append(mfunc(x))
        adj_diff.append(np.linalg.norm(x - x_old))

    return x, fval, adj_diff

sigmoid = lambda z: 1 / (1 + jnp.exp(-z))

"Obtain maximum a-posteriori estimate (mode of distribution)"
def obtain_MAP(X_train, y_train, lam):
    print("Starting MAP computation")
    Af = lambda x: X_train @ x
    As = lambda x: X_train.T @ x
    prox = lambda x, tau: np.maximum(np.abs(x)-tau, 0)*np.sign(x)
    #mfunc = lambda x: np.sum(np.log(1 + np.exp(-y_train * Af(x)))) + lam*np.linalg.norm(x,ord=1)
    mfunc = lambda x: np.linalg.norm(x,ord=1)
    #tau = 1/20 #stepsize
    #tau = 1/ 50
    #tau = tau_prox
    tau = 1 /  Lf
    nIter =10000
    dG = lambda x: As(Af(x) - y_train)
    proxF = lambda x,tau: prox(x,tau*lam)
    xinit = As(y_train)

    #run restarted fista
    x_mode,fval,x_adj = rFISTA(proxF, dG, tau, xinit,nIter,mfunc)
    #x_mode,fval, x_adj = ISTA(proxF, dG, tau, xinit,nIter,mfunc)

    return x_mode,fval,x_adj

def binary_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def lasso_cv_custom(X, y, lam_vals, n_folds=10, random_seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    mean_accuracies = []

    for lam in lam_vals:
        fold_accuracies = []

        print(f"Starting kfold for lambda = {lam}")

        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Run your custom FISTA-based MAP solver
            x_mode, _, _ = obtain_MAP(X_train_fold, y_train_fold, lam)

            # Predict and compute accuracy on validation fold
            logit_cv = X_val_fold @ x_mode  # Predicted log-odds
            #y_pred_class = (prob_pred > 0.5).astype(np.int32)  # Convert to class labels using 0.5 threshold
            #y_pred_class = np.sign(prob_pred)
            prob_cv = 1 / (1 + np.exp(-logit_cv))
            y_pred_class = (prob_cv >= 0.5).astype(np.int32)  # Convert to class labels
            
            
            y_val_adjusted = (y_val_fold == 1).astype(int)


            #accuracy = accuracy_score(y_val_fold, y_pred_class)
            accuracy = binary_accuracy(y_val_adjusted, y_pred_class)
            fold_accuracies.append(accuracy)

        mean_cv_accuracy = np.mean(fold_accuracies)
        mean_accuracies.append(mean_cv_accuracy)
        print(f"Lambda = {lam:.4f}, CV Accuracy = {mean_cv_accuracy:.4f}")

    print(mean_accuracies)

    best_idx = np.argmax(mean_accuracies)
    best_lambda = lam_vals[best_idx]
    print(f"\nBest lambda = {best_lambda:.4f} with average CV Accuracy = {mean_accuracies[best_idx]:.4f}")
    return best_lambda, mean_accuracies





def get_data():
    dim = 3
    db_bounds = [-1,2,3]
    bias = -3


    def in_region(x):
        # extend linear d.b. to arbitrary dimension
        return (np.dot(db_bounds, x) + bias)


    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    def generate_data(n, dim, noise_dim, noisy_var, bounds):
        X = np.zeros((n,dim))
        
        
        #generate X_i at random from the rectangle [b0, b1] x [b2, b3]
        for i in range(n):
            for d in range(dim):
                X[i,d] = rnd.uniform(bounds[d,0],bounds[d,1])
            #X[i,0] = rnd.uniform(bounds[0],bounds[1])
            #X[i,1] = rnd.uniform(bounds[2],bounds[3])
            
        # Y takes value 0 or 1
        Y = np.zeros(n)
        print(X[0,:])
        for i in range(n):
            y = in_region(X[i,:])
            Y[i] = rnd.binomial(1,sigmoid(0.5*y))
            

        #Add quadratic terms
        #X = np.hstack((X, X**2))
        
        #Add noisy, useless dimensions    
        
        #Use code below for uncorrelated random noise
        noise = np.random.normal(0, noisy_var, size=(n, noise_dim))  # Gaussian noise
        
        #Use this code for correlated random noise
        #mean = np.zeros(noise_dim)
        #rho = 0.5
        #cov = noisy_var * (rho * np.ones((noise_dim, noise_dim)) + (1-rho) * np.eye(noise_dim))
        #noise = np.random.multivariate_normal(mean, cov, size=n)
        
        
        
        X_full = np.hstack((X, noise))  # Combine original data with noise
        
        Y = 2 * Y - 1
        
        return X_full,Y

    n = 200 #number of data points
    #n = 50
    #noise_d = 28

    bounds = np.array([[-2,3],[-4,5],[-1,2]])
    noise_d = 97
    noisy_var = 10
    X_train_unscaled,y_train = generate_data(n,dim,noise_d,noisy_var,bounds)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_unscaled)
        
    #blues = X_train[y_train==1,:]
    #reds = X_train[y_train==0,:]

    n_test = 50

    X_test_unscaled,y_test = generate_data(n_test,dim,noise_d,noisy_var,bounds)
    X_test = scaler.transform(X_test_unscaled)

    # Adjust y_test from {-1, +1} to {0, 1} for compatibility between loss function and metrics
    y_test_adjusted = (y_test == 1).astype(int)
    
    
    return X_train, y_train, X_test, y_test, y_test_adjusted



#for i in range(dim):
#    strname = 'X' + str(i + 1)
#    feature_names.append(strname)
    
#for i in range(noise_d):
#    strname = 'N' + str(i + 1)
#    feature_names.append(strname)


regen = 100

lamlen = 10
betalen = 1




method_base =  ['PROXL1',
                #'EULA',
                #'EULA_FB',
                'uv_FB',
                'cart_BOB',
                'cart_PEM'
                #'uv_Bessel',
                #'MASCIR',
                #'one_part_mala',
                #'one_part_uvfb',
                #'one_part_eula',
                #'one_part_bessel'
                #'one_part_gibbs',
                ]
n_methods = len(method_base)



#Parameters for assessing mixing
#burn_in_base = 0
#niter_base = 10**6
#subsamp_base = 1

#T_burn = 100

niter_base = 10**5
#burn_in_base = 10**5
burn_in_base = 2*10**4
subsamp_base = 100

#niter_base = 3*10**5
#burn_in_base = 10**5
#subsamp_base = 1000

#taus = [4/Lf]*len(method_base)

all_accs = np.zeros((n_methods,regen))
all_rocauc = np.zeros((n_methods,regen))

all_deter_acc = np.zeros(regen)
all_deter_rocauc = np.zeros(regen)


for bigi in range(regen):
    X_train, y_train, X_test, y_test, y_test_adjusted = get_data()
    
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    
    "--- DEFINE BETA AND LAMBDA RANGES HERE ---"
    
    
    
    betavals = [1]
    maxlam = 0.9*jnp.max(jnp.abs(X_train.T@y_train))
    lamvals = np.logspace(-1,np.log10(maxlam),lamlen)
    
    
    
    
    m, n = X_train.shape
    lam_arr = (0.1)*jnp.max(jnp.abs(X_train.T@y_train))
    lam = lam_arr.item()
    M = 1
    #Lf = 3000
    #Lf = np.linalg.norm(X_train)**2
    Lf = np.linalg.norm(X_train,2)**2
    gamma = 1 / (Lf)
    tau_prox = 1 / (30*(Lf + 1/gamma))
    #tau_prox = 0.01
    beta = 3000
    key = random.key(0)
    
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train.flatten())
    
    taus = [4/Lf, 1/(2*Lf), 1/(2*Lf), 1/(2*Lf)]
    
    """
    
    # Example usage: tuning lambda over a grid
    best_lam, accuracy_vals = lasso_cv_custom(X_train, y_train, lamvals)
    
    # Final evaluation on the test set using best lambda
    x_mode_cv, _, _ = obtain_MAP(X_train, y_train, best_lam)
    prob_pred_test = X_test @ x_mode_cv.T
    logits_tst = 1 / (1 + np.exp(-prob_pred_test))
    y_test_pred = (prob_pred_test >= 0.5).astype(int)  # Convert to class labels
    #y_test_pred = np.sign(prob_pred_test)
    #test_accuracy = accuracy_score(y_test_adjusted, y_test_pred)
    test_accuracy = binary_accuracy(y_test_adjusted, y_test_pred)
    rocauc_map = roc_auc_score(y_test_adjusted, logits_tst)
    
    print(f"Test Accuracy using MAP with λ={best_lam}: {test_accuracy:.4f}")
    print(f"Test ROC/AUC using MAP with λ={best_lam}: {rocauc_map:.4f}")
    
    all_deter_acc[bigi] = test_accuracy
    all_deter_rocauc[bigi] = rocauc_map
    
    """
    
    mesh_rocauc = np.zeros((n_methods,len(lamvals),len(betavals)))
    
    mesh_acc = np.zeros((n_methods,len(lamvals),len(betavals)))
    
    mesh_lasso_acc = np.zeros((len(lamvals),len(betavals)))
    mesh_lasso_rocauc = np.zeros((len(lamvals),len(betavals)))
    
        
    for la in range(len(lamvals)):
        lam = lamvals[la]
        
        "--- REGENERATE MAP (FOR CHECKING)---"
        x_mode,fval,x_adj = obtain_MAP(X_train, y_train, lam)
    
        # Compute logits for this batch
        logits_MAP = X_test @ x_mode.T  # Shape: (n_test, 1)
    
        # Apply sigmoid to compute probabilities
        predictions_MAP = 1 / (1 + np.exp(-logits_MAP))  # Shape: (n_test, 1)
    
        binary_MAP = (predictions_MAP >= 0.5).astype(int)
        
        
        
        
        for be in range(len(betavals)):
            beta = betavals[be]
            
            "--- GENERATE SUBDIRECTORY --- "
            subdir = f"lambda = {lam:.3f}, beta = {beta:.3f}"
            print(subdir)
            save_sub = os.path.join("SIMULATION","BINARY","SYNTH","BELAM",subdir)
            os.makedirs(save_sub, exist_ok=True)
            
            
            save_dist_str = "barplot_MAP.pdf"
    
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(x_mode)), x_mode)
            plt.xticks(rotation=90)  # Rotate feature names for readability
            plt.ylabel("Weight Magnitude")
            plt.title("Feature Importance for MAP")
            plt.grid()
            fig_path_bar = os.path.join(save_sub,save_dist_str)
            plt.savefig(fig_path_bar, dpi=300, bbox_inches='tight')  # Save with high resolution
            plt.show()
            
            
            
            
            
            "--- SETUP NEW PARAMS FOR SIM"
            setup_base = setup_data(X_train,n,lam,y_train,M,gamma,beta,key)
            
            
            "--- SIMULATE ---"
            
            runout, burn_time, sample_time = Runner(setup_base, method_base).runner(niter_base, taus, burn_in_base, subsamp_base) 
            savefile = os.path.join(save_sub,"data_compressed.npz")
            np.savez_compressed(savefile, runout=runout)
            
            """
            
            "--- CHECK ESS ---"
    
    
    
            min_ess_vals = []
            max_ess_vals = []
            mean_ess_vals = []
            for k in range(len(method_base)):
                ess_values = np.array([az.ess(runout[k,:,0,j]) for j in range(M)])
                print(f"Min ESS for {method_base[k]}: {ess_values.min():.2f}, Max ESS: {ess_values.max():.2f}")
                min_ess_vals.append(np.min(ess_values))
                max_ess_vals.append(np.max(ess_values))
                mean_ess_vals.append(np.mean(ess_values))
    
            plt.figure()
                
            x = np.arange(len(method_base))  # X positions for bars
            width = 0.3  # Width of the bars
    
            save_dist_str = "ESS.pdf"
    
            fig, ax = plt.subplots(figsize=(8, 5))
    
            # Plot bars for min and max ESS
            ax.bar(x - width, min_ess_vals, width, label="Min ESS", color="blue", alpha=0.7)
            ax.bar(x , mean_ess_vals, width, label="Mean ESS", color="green", alpha=0.7)
            ax.bar(x + width, max_ess_vals, width, label="Max ESS", color="red", alpha=0.7)
    
            # Labels and title
            ax.set_xlabel("Methods")
            ax.set_ylabel("ESS")
            ax.set_title("Min, Mean and Max ESS for Different Methods")
            ax.set_xticks(x)
            ax.set_xticklabels(method_base)  # Set method names as x-axis labels
            ax.legend()
            #ax.set_ylim(0, niter // subsamp)
            fig_path_bar = os.path.join(save_sub,save_dist_str)
            plt.savefig(fig_path_bar, format="pdf",bbox_inches="tight")
            plt.show()
    
    
    
    
            "--- PLOT TIMES"
    
            sum_burn_time = np.sum(burn_time, axis = 1)
            sum_samp_time = np.sum(sample_time, axis = 1)
    
            "Create a bar-chart of mean per-iteration times"
            filename_bar = "times_bar.pdf"
            fig_path4 = os.path.join(save_sub,filename_bar)
            time_values = (sum_burn_time+sum_samp_time) / (burn_in_base+niter_base)
            'Create a value-dependent colour scheme'
            norm = plt.Normalize(min(time_values), max(time_values))
            color_bar = sns.color_palette("Blues", as_cmap=True)(norm(time_values))
    
            sns.barplot(x=method_base, y=time_values, palette=color_bar)
    
            plt.xlabel("Numerical Method")
            plt.ylabel("t(s)")
            plt.title("Mean time per iteration, per stepsize")
            plt.savefig(fig_path4, dpi=300, bbox_inches='tight')  # Save with high resolution
            plt.show()
    
            """
    
    
    
    
            #Firstly, check that chain is well-mixed at start of iterations
    
            
            n_samp = runout[0,:,0,0].shape[0]
            n_test = X_test.shape[0]
            #predictions = np.zeros((len(method_base),n_samp, n_test))
    
            
            """
            acc_list = []  # To store mean classification errors for each method
            
            # Loop over methods
            for i in range(n_methods):
                samples = runout[i]
                acc_method_list = []  # Store mean classification error for each posterior sample set
                probs_all = []
                # Process weights in chunks
                for s in range(n_samp):  # For each subsampled iteration
                    particles = []  # Store the mean classification error for each particle
                    
                    for p in range(M):  # For each posterior sample from the method
                        beta = samples[s, :, p]  # shape: (x_dim,)
                        
                        # Compute logits: X_test @ beta (this gives log-odds)
                        logits = X_test @ beta  # shape: (n_test,)
                        
                        # Compute predicted probabilities using sigmoid function
                        prob = 1 / (1 + jnp.exp(-logits))  # shape: (n_test,)
                        
                        probs_all.append(prob)
                        
                        # Classify based on 0.5 threshold
                        y_pred_class = (prob > 0.5).astype(jnp.float32)  # shape: (n_test,)
                        
                        acc = jnp.mean(y_pred_class == y_test_adjusted)
                        particles.append(acc)
                    
                    # Compute the mean classification acc for this subsample s
                    acc_method_list.append(jnp.array(particles))  # shape: scalar, mean classification acc for this sample
            
                # Compute the mean classification accuracy across all subsampled iterations (s = 1...n_samp)
                acc_list.append(jnp.mean(jnp.array(acc_method_list)))  # Store for this method
                
                probs_all = jnp.stack(probs_all, axis=0)
                y_prob = jnp.mean(probs_all, axis=0)
                roc_auc = roc_auc_score(y_test_adjusted, np.array(y_prob))
                mesh_rocauc[i,la,be] = roc_auc
            
            # Final MCE array for all algorithms: shape (no_algs,)
            mesh_acc[:, la, be] = np.array(acc_list)  # Store for plotting or analysis
            """
            
            
            acc_list = []
            for i in range(n_methods):
                samples = runout[i]  # shape: (n_samp, x_dim, 1)
                acc_method_list = []
                probs_all = []
            
                for s in range(n_samp):
                    beta = samples[s, :, 0]  # shape: (x_dim,)
                    logits = X_test @ beta   # shape: (n_test,)
                    prob = 1 / (1 + jnp.exp(-logits))  # sigmoid
                    probs_all.append(prob)
            
                    y_pred_class = (prob > 0.5).astype(jnp.float32)
                    acc = jnp.mean(y_pred_class == y_test_adjusted)
                    acc_method_list.append(acc)
            
                # Stack predictions and compute average accuracy
                acc_array = jnp.stack(acc_method_list)  # shape: (n_samp,)
                acc_list.append(jnp.mean(acc_array))
            
                # Stack all probs and compute predictive mean
                probs_all = jnp.stack(probs_all, axis=0)  # shape: (n_samp, n_test)
                y_prob = jnp.mean(probs_all, axis=0)  # shape: (n_test,)
            
                roc_auc = roc_auc_score(y_test_adjusted, np.array(y_prob))  # send to numpy once
                mesh_rocauc[i, la, be] = roc_auc
            
            mesh_acc[:, la, be] = np.array(acc_list)
            
                
                
            helpr = Helper(setup_base)
            X_test_fun = lambda x: X_test @ x
            #X_test_s_fun = lambda x: X_test.T @ x
    
            #ytst_M = jnp.tile( y_test[:,None],(1,M))
                
            accuracy_MAP = binary_accuracy(y_test_adjusted, binary_MAP)
            roc_auc_MAP = roc_auc_score(y_test_adjusted, predictions_MAP)
            
            mesh_lasso_acc[la,be] = accuracy_MAP
            mesh_lasso_rocauc[la,be] = roc_auc_MAP
    
            
            output_file = os.path.join(save_sub, "results.txt")
            with open(output_file, "w") as f:
                print(f"Results for MAP")
                print(f"Test Accuracy: {accuracy_MAP:.4f}")
                print(f"Test ROC-AUC: {roc_auc_MAP:.4f}")
    
                f.write(f"Results for MAP\n")
                f.write(f"Test Accuracy: {accuracy_MAP:.4f}\n")
                f.write(f"Test ROC-AUC: {roc_auc_MAP:.4f}\n")
                
                for i in range(len(method_base)):
                    # Evaluate performance
                   
                    
                    
                    our_logloss = helpr.objfun(np.mean(runout[i,:,:,:], axis = (0,2)).reshape(-1,1),X_test_fun,y_test)[0]
                    print(f"Results for {method_base[i]}")
                    print(f"Mean Test Accuracy: {mesh_acc[i, la, be]}")
                    print(f"Test ROC-AUC: {roc_auc:.4f}")
                    
                    f.write(f"Results for {method_base[i]}\n")
                    f.write(f"Median Mean Classification Error: {mesh_acc[i, la, be]}\n")
                    f.write(f"Test ROC-AUC: {roc_auc:.4f}\n")
                    
    
    
    max_idx_lasso = np.unravel_index(np.argmax(mesh_lasso_acc), mesh_lasso_acc.shape)
    lam_idx_l, beta_idx_l = max_idx_lasso
    
    all_deter_acc[bigi] = mesh_lasso_acc[lam_idx_l, beta_idx_l]
    all_deter_rocauc[bigi] = mesh_lasso_rocauc[lam_idx_l, beta_idx_l]
    
    best_lam_lasso = lamvals[lam_idx_l]
    
    
    # Prepare arrays to store best values
    best_lam = np.zeros(n_methods)
    best_beta = np.zeros(n_methods)
    
    for i in range(n_methods):
        # Find index of max acc for method i
        max_idx = np.unravel_index(np.argmax(mesh_acc[i]), mesh_acc[i].shape)
        lam_idx, beta_idx = max_idx
    
        all_accs[i,bigi] = mesh_acc[i, lam_idx, beta_idx]
        all_rocauc[i,bigi] = mesh_rocauc[i,lam_idx,beta_idx]
        
        best_lam[i] = lamvals[lam_idx]
        best_beta[i] = betavals[beta_idx]
     
    print("\n\n")  
     
    
    # Optional: print results
    print(f"Lasso: Best mean test accuracy: {all_deter_acc[bigi]:.4f}, with ROC/AUC = {all_deter_rocauc[bigi]} at λ={best_lam_lasso}")
    for i in range(n_methods):
        print(f"Method {method_base[i]}: Best mean test accuracy = {all_accs[i,bigi]:.4f}, with mean ROC-AUC {all_rocauc[i,bigi]:.4f} at λ = {best_lam[i]:.4f}, β = {best_beta[i]:.4f}")
        
print("\n\n")

for i in range(n_methods):
    median_mean_acc = np.median(all_accs[i])
    median_rocauc = np.median(all_rocauc[i])
    print(f"Method {method_base[i]}: Median mean test accuracy = {median_mean_acc:.4f}, with median mean ROC-AUC {median_rocauc:.4f}")
    
lasso_median_acc = np.median(all_deter_acc)
lasso_median_roc = np.median(all_deter_rocauc)
print(f"Lasso median mean test accuracy = {lasso_median_acc:.4f}, with median mean ROC-AUC {lasso_median_roc:.4f}")