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
from sklearn.metrics import mean_squared_error


save_xtra = os.path.join("SIMULATION","MMSE","BELAM")
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


# Set parameters
reload_var = False
#train_size = 10000
#test_size = 2000

# File path to save the dataset locally
file_path = os.path.join(save_xtra, "synth_data.npz")



def generate_data_example_1(m, n, sigma, beta, key):
    if n % 4 != 0:
        raise ValueError("In this example, need feature no divisible by 4.")
    
    x0 = np.concatenate([np.zeros(n//4), 2*np.ones(n//4), np.zeros(n//4), 2*np.ones(n//4)])
    
    # Create the covariance matrix with 0.95 correlation
    rho = 0.95
    cov = rho * jnp.ones((n, n)) + (1 - rho) * jnp.eye(n)

    # Step 2: Cholesky decomposition
    L = jnp.linalg.cholesky(cov)

    # Step 3: Generate standard normal samples
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(m, n))  # z ~ N(0, I)

    # Step 4: Apply transformation
    X = z @ L.T  # X ~ N(0, cov)

    epsvec = sigma * jax.random.normal(key, shape=(m,))

    y = X@x0 + epsvec
    
    return X,y



def generate_data_example_3(m, x0, sigma, beta, key):
    n = len(x0)
    
    # Create the covariance matrix with 0.95 correlation
    rho = 0.95
    cov = rho * jnp.ones((n, n)) + (1 - rho) * jnp.eye(n)

    # Step 2: Cholesky decomposition
    L = jnp.linalg.cholesky(cov)

    # Step 3: Generate standard normal samples
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(m, n))  # z ~ N(0, I)

    # Step 4: Apply transformation
    X = z @ L.T  # X ~ N(0, cov)

    epsvec = sigma * jax.random.normal(key, shape=(m,))

    y = X@x0 + epsvec
    
    return X,y


def generate_data_example_4(m, n, sigma, beta, key):
    q = 10
    x0 = np.concatenate([5 * np.ones(q), np.zeros(n - q)])
    
    # Create the covariance matrix with 0.95 correlation
    rho = 0.95
    cov = rho * jnp.ones((n, n)) + (1 - rho) * jnp.eye(n)

    # Step 2: Cholesky decomposition
    L = jnp.linalg.cholesky(cov)

    # Step 3: Generate standard normal samples
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(m, n))  # z ~ N(0, I)

    # Step 4: Apply transformation
    X = z @ L.T  # X ~ N(0, cov)

    epsvec = sigma * jax.random.normal(key, shape=(m,))

    y = X@x0 + epsvec
    
    return X,y


feature_no = 40
master_key = jax.random.PRNGKey(0)
key_train, key_test = jax.random.split(master_key)

n = 100 #number of data points
#noise_d = 28

x0 = jnp.array([3,1.5,0,0,2,0,0,0])

noisy_sig = 25
beta = 10

#X_train,y_train = generate_data_example_3(n, x0, noisy_sig, beta, key_train)
X_train,y_train = generate_data_example_1(n, feature_no, noisy_sig, beta, key_train)


n_test = 400


#X_test,y_test = generate_data_example_3(n_test, x0, noisy_sig, beta, key_test)
X_test,y_test = generate_data_example_1(n_test, feature_no, noisy_sig, beta, key_test)

feature_names = []

for i in range(feature_no):
    strname = 'X' + str(i + 1)
    feature_names.append(strname)


print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


"--- DEFINE BETA AND LAMBDA RANGES HERE ---"


betavals = np.logspace(0,np.log10(50),6)
maxlam = 0.5*jnp.max(jnp.abs(X_train.T@y_train))
lamvals = np.logspace(-1,np.log10(maxlam),10)
#lamvals = np.linspace(1,maxlam,6)

#betavals= [1, 100]
#lamvals = [1,100]




m, n = X_train.shape
lam_arr = (0.1)*jnp.max(jnp.abs(X_train.T@y_train))
lam = lam_arr.item()
M = 10
#Lf = 3000
#Lf = np.linalg.norm(X_train)**2
Lf = np.linalg.norm(X_train,2)**2
gamma = 1 / (Lf)
tau_prox = 1 / (30*(Lf + 1/gamma))
#tau_prox = 0.01

key = random.key(0)

X_train = jnp.array(X_train)
y_train = jnp.array(y_train.flatten())

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
            #print(f"Final consecutive difference for restarted FISTA: {np.linalg.norm(xkm-x)} at iteration {k}")
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


def lasso_cv_custom(X, y, lam_vals, n_folds=10, random_seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    mean_errors = []

    for lam in lam_vals:
        fold_errors = []
        
        print(f"Starting kfold for lambda = {lam}")

        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Run your custom FISTA-based MAP solver (cleaner call!)
            x_mode, _, _ = obtain_MAP(X_train_fold, y_train_fold, lam)

            # Predict and compute MSE on validation fold
            y_pred = X_val_fold @ x_mode
            fold_errors.append(mean_squared_error(y_val_fold, y_pred))

        mean_cv_error = np.mean(fold_errors)
        mean_errors.append(mean_cv_error)
        print(f"Lambda = {lam:.4f}, CV MSE = {mean_cv_error:.4f}")

    best_idx = np.argmin(mean_errors)
    best_lambda = lam_vals[best_idx]
    print(f"\nBest lambda = {best_lambda:.4f} with average CV MSE = {mean_errors[best_idx]:.4f}")
    return best_lambda, mean_errors


#lam_vals = np.logspace(-2, np.log10(maxlam), 20)  # e.g., from 0.01 to 100
best_lam, mse_vals = lasso_cv_custom(X_train, y_train, lamvals)

x_mode, _, _ = obtain_MAP(X_train, y_train, best_lam)
y_test_pred = X_test @ x_mode
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE using MAP with λ={best_lam}: {test_mse:.4f}")


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

#niter_base = 3*10**6
#burn_in_base = 10**6
#subsamp_base = 10000

niter_base = 3*10**5
burn_in_base = 10**5
subsamp_base = 1000

taus = [4/Lf]*len(method_base)

mesh_mmse = np.zeros((n_methods,len(lamvals),len(betavals)))

for la in range(len(lamvals)):
    lam = lamvals[la]
    
    
    
    for be in range(len(betavals)):
        beta = betavals[be]
        
        "--- GENERATE SUBDIRECTORY --- "
        subdir = f"lambda = {lam:.0f}, beta = {beta:.0f}"
        print(subdir)
        save_sub = os.path.join("SIMULATION","MMSE","8dim","BELAM",subdir)
        os.makedirs(save_sub, exist_ok=True)
        
        
        
        
        
        "--- SETUP NEW PARAMS FOR SIM"
        setup_base = setup_data(X_train,n,lam,y_train,M,gamma,beta,key)
        
        
        "--- SIMULATE ---"
        
        runout, burn_time, sample_time = Runner(setup_base, method_base).runner(niter_base, taus, burn_in_base, subsamp_base) 
        savefile = os.path.join(save_sub,"data_compressed.npz")
        np.savez_compressed(savefile, runout=runout)
        
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






        #Firstly, check that chain is well-mixed at start of iterations

        
        n_samp = runout[0,:,0,0].shape[0]
        n_test = X_test.shape[0]
        #predictions = np.zeros((len(method_base),n_samp, n_test))

        batch_size = 1000  # Number of samples to process at a time


        mmse_list = []
        # Loop over methods
        for i in range(n_methods):
            #method_predictions = np.zeros((n_samp, n_test))  # Temporary storage for this method's predictions
            samples = runout[i]
            pred_list = []
            # Process weights in chunks
            for s in range(n_samp):
                pred_particles = []
                for p in range(M):
                    beta = samples[s, :, p]  # shape: (x_dim,)
                    y_pred = X_test @ beta   # shape: (n_test,)
                    pred_particles.append(y_pred)
                pred_particles = jnp.stack(pred_particles, axis=0)  # shape: (P, n_test)
                pred_list.append(pred_particles)
        
            preds = jnp.stack(pred_list, axis=0)  # shape: (S, P, n_test)
        
            # Step 3: Compute posterior mean prediction
            y_pred_mean = jnp.mean(preds, axis=(0, 1))  # shape: (n_test,)
        
            # Step 4: Compute MMSE for this algorithm
            mmse = jnp.mean((y_pred_mean - y_test)**2)
            mmse_list.append(mmse)
        
        # Final MMSE array for all algorithms: shape (no_algs,)
        mesh_mmse[:,la,be] = np.array(mmse_list)
        #mmse_all = jnp.array(mmse_list)



        "--- SAMPLE UNCERTAINTY-RELATED RESULTS --- "
        "Compute Uncertainty from Your Posterior Samples"
        mean_image = np.mean(runout, axis=(1,3))
        std_image = np.std(runout, axis=(1,3))


        "--- MIXING GRAPH ---"

        filename = "mix_check.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            #method_diff = np.diff(runout[i,:,:,0], axis=0)
            method_norms = np.linalg.norm(runout[i,:,:,0], axis=1)
            plt.loglog(method_norms, label = method_base[i])
            plt.xlabel("Burn-in")  # Label for x-axis
            plt.ylabel("Error")  # Label for y-axis
            plt.legend()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
            
        fig, axes = plt.subplots(2, n_methods, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)

        filename = "weight_many_bars.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            
            # Mean weights
            axes[0, i].bar(range(len(mean_image[i])), mean_image[i], tick_label=feature_names)
            axes[0, i].set_title(f"{method_base[i]} - Mean Weights")
            
            
            # Quantile differences
            # Compute log-difference
            quantile_95_method = np.percentile(runout[i, :, :, 0], 95, axis=0)
            quantile_5_method = np.percentile(runout[i, :, :, 0], 5, axis=0)
            quantile_diff_method = np.log(np.abs(quantile_95_method - quantile_5_method))
            
            axes[1, i].bar(range(len(quantile_diff_method)), quantile_diff_method, tick_label=feature_names)
            axes[1, i].set_title(f"{method_base[i]} - Log-Quantile diffs")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
            
        
        
        save_dist_str = "weight_means.pdf"
        fig_path_bar = os.path.join(save_sub, save_dist_str)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        for i in range(n_methods):
            ttl_str = method_base[i] + ' - Mean Weights per Feature'
        
            axes[i].bar(range(len(mean_image[i])), mean_image[i], tick_label=feature_names)
            axes[i].set_xticklabels(feature_names, rotation=45, ha="right",rotation_mode="anchor")
            axes[i].set_ylabel("Mean Weight Value")
            axes[i].set_title(ttl_str)
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(fig_path_bar, format="pdf", bbox_inches="tight")
        plt.show()
        
        save_dist_str = "weight_mean_map.pdf"
        fig_path_bar2 = os.path.join(save_sub,save_dist_str)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        
        for i in range(n_methods):
            ttl_str = method_base[i] + ' - Absolute Difference from MAP Estimate'
            abs_diff = np.abs(mean_image[i] - x_mode)  # Absolute difference per feature
        
            axes[i].bar(range(len(abs_diff)), abs_diff, tick_label=feature_names)
            axes[i].set_xticklabels(feature_names, rotation=45, ha="right",rotation_mode="anchor")
            axes[i].set_ylabel("Absolute Difference")
            axes[i].set_title(ttl_str)
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(fig_path_bar2, format="pdf", bbox_inches="tight")
        plt.show()
        

        # Calculate the 95th and 5th quantiles for each pixel 
        save_dist_str = "weight_quantile.pdf"
        fig_path_bar3 = os.path.join(save_sub, save_dist_str)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        
        for i in range(n_methods):
            ttl_str = method_base[i] + ' - 95 percent Credibility Interval'
            # Compute log-difference
            quantile_95_method = np.percentile(runout[i, :, :, 0], 97.5, axis=0)
            quantile_5_method = np.percentile(runout[i, :, :, 0], 2.5, axis=0)
            quantile_diff_method = np.log(np.abs(quantile_95_method - quantile_5_method))
        
            axes[i].bar(range(len(quantile_diff_method)), quantile_diff_method, tick_label=feature_names)
            axes[i].set_xticklabels(feature_names, rotation=45, ha="right",rotation_mode="anchor")
            axes[i].set_ylabel("Log |97.5th - 2.5th Quantile|")
            axes[i].set_title(ttl_str)
            axes[i].grid()
        plt.tight_layout()
        plt.savefig(fig_path_bar3, format="pdf", bbox_inches="tight")
        plt.show()

            
            
            
            
        helpr = Helper(setup_base)
        X_test_fun = lambda x: X_test @ x
        #X_test_s_fun = lambda x: X_test.T @ x

        #ytst_M = jnp.tile( y_test[:,None],(1,M))
            
    
        
       
        
        output_file = os.path.join(save_sub, "results.txt")
        with open(output_file, "w") as f:
            for i in range(len(method_base)):
                mean_norm_diff = np.linalg.norm(mean_image[i] - x_mode)
                # Evaluate performance
        
                our_logloss = helpr.objfun(np.mean(runout[i,:,:,:], axis = (0,2)).reshape(-1,1),X_test_fun,y_test)[0]
                print(f"Results for {method_base[i]}")
                print(f"MMSE: {mesh_mmse[i,la,be]:.4f}")
                f.write(f"Results for {method_base[i]}\n")
                f.write(f"MMSE: {mesh_mmse[i,la,be]:.4f}\n")



# Prepare arrays to store best values
best_mmse = np.zeros(n_methods)
best_lam = np.zeros(n_methods)
best_beta = np.zeros(n_methods)

for i in range(n_methods):
    # Find index of min MMSE for method i
    min_idx = np.unravel_index(np.argmin(mesh_mmse[i]), mesh_mmse[i].shape)
    lam_idx, beta_idx = min_idx

    best_mmse[i] = mesh_mmse[i, lam_idx, beta_idx]
    best_lam[i] = lamvals[lam_idx]
    best_beta[i] = betavals[beta_idx]

# Optional: print results
for i in range(n_methods):
    print(f"Method {method_base[i]}: Best MMSE = {best_mmse[i]:.4f} at λ = {best_lam[i]:.4f}, β = {best_beta[i]:.4f}")

# Create meshgrid
Beta, Lambda = np.meshgrid(betavals, lamvals, indexing='ij')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "mmse_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    color_map = ax.pcolormesh(Lambda, Beta, mesh_mmse[i].T, cmap="coolwarm", shading='auto')
    
    # Colorbar for accuracy
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Test MMSE")

    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"{method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()



