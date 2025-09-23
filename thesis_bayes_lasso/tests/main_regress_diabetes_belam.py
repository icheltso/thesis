# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:09:50 2024

@author: ichel
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
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

from sklearn.datasets import load_diabetes
from sklearn.base import BaseEstimator, RegressorMixin


save_xtra = os.path.join("SIMULATION","REGRESS","DIABETES","BELAM")
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
    loss_type = "square_loss"
    #loss_type = "log_loss"
    
    setup_data = Setup(sigma, loss_type, Af, As, n, lam, y, y2, initx, initv, initz, M, gamma, R)
    
    #np.savez(file_path, A=A, As=As, n=n, lam=lam, x0=x0, y=y, y2=y2, initx=initx, initv=initv, M=M, gamma=gamma)
    #with open(file_path, 'wb') as file:
    #    dill.dump(todump, file)
        
    return setup_data

def running_mean(samples):
    n_iter, n_params = samples.shape
    running_means = np.zeros((n_iter, n_params))

    cumulative_sum = np.zeros(n_params)
    for t in range(n_iter):
        cumulative_sum += samples[t]
        running_means[t] = cumulative_sum / (t + 1)

    return running_means

"Compute acf for a given method, at a given lag, for a given weight."
def get_acf(lag, method_id, weight_id):
    method_idx = method_id
    weight_idx = weight_id

    # Extract the sampled values for the weight
    weight_samples = runout[method_idx, :, weight_idx, 0]  # Shape: (n_samples,)

    # Compute ACF
    lag_acf = acf(weight_samples, nlags=lag, fft=True)  # nlags: max lags to compute

    # Plot ACF
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lag_acf)), lag_acf, marker='o', linestyle='-', color='b')
    plt.axhline(y=0, color='k', linestyle='--')  # Add a horizontal line at 0
    plt.title(f"Autocorrelation Function for Weight {weight_idx}")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid()
    plt.show()
    
def get_trace(method_id, weight_id):
    # Example: trace plot for a single weight (from first method)
    weight_samples = runout[method_id, :, weight_id, 0]  # Shape: (n_samples,)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(weight_samples)
    plt.title("Trace Plot for Weight")
    plt.xlabel("Iterations")
    plt.ylabel("Weight Value")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
def geweke_z(chain, first=0.1, last=0.5):
    """
    Compute Geweke Z-score for a 1D array (a single chain of a parameter).
    """
    n = len(chain)
    first_slice = chain[:int(first * n)]
    last_slice = chain[-int(last * n):]

    mean_first = np.mean(first_slice)
    mean_last = np.mean(last_slice)

    var_first = np.var(first_slice, ddof=1)
    var_last = np.var(last_slice, ddof=1)

    z = (mean_first - mean_last) / np.sqrt(var_first/len(first_slice) + var_last/len(last_slice))
    return z
    
    
    


# Set parameters
reload_var = False
#train_size = 10000
#test_size = 2000


# Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
columns=diabetes.feature_names
        
X_train_0, X_test_0, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_0)
X_test = scaler.transform(X_test_0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


"--- DEFINE BETA AND LAMBDA RANGES HERE ---"


#betavals = np.logspace(0,np.log10(3000),6)
maxlam = 0.5*jnp.max(jnp.abs(X_train.T@y_train))
lamvals = np.logspace(0,np.log10(maxlam),5)
#lamvals = np.linspace(1,maxlam,6)

betavals= [0.001, 1]
#lamvals = [1]



#num_points = 6  # Number of points
#x = np.linspace(1, num_points, num=num_points)  # Evenly spaced values
#poly_growth = (x / num_points)**0.5  # Polynomial growth
#poly_growth = (poly_growth - poly_growth.min()) / (poly_growth.max() - poly_growth.min())  # Normalize to [0, 1]
#poly_growth = poly_growth * (maxlam - 1) + 1  # Scale to start at 1 and end at n

#lamvals = [maxlam]



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
#beta = 3000
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

        if np.linalg.norm(xkm-x)<tol:
            print(f"Final consecutive difference for restarted FISTA: {np.linalg.norm(xkm-x)} at iteration {k}")
            break
        
    print(f"Final consecutive difference for restarted FISTA: {np.linalg.norm(xkm-x)}")
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
def obtain_MAP(lam):
    print("Starting MAP computation")
    Af = lambda x: X_train @ x
    As = lambda x: X_train.T @ x
    prox = lambda x, tau: np.maximum(np.abs(x)-tau, 0)*np.sign(x)
    #mfunc = lambda x: np.sum(np.log(1 + np.exp(-y_train * Af(x)))) + lam*np.linalg.norm(x,ord=1)
    mfunc = lambda x: np.linalg.norm(x,ord=1)
    #tau = 1/20 #stepsize
    #tau = 1/ 50
    #tau = tau_prox
    tau = 4 / (10*Lf)
    nIter =10000
    dG = lambda x: As(Af(x)-y_train)
    #dG = lambda x: As(-sigmoid(-y_train * Af(x)) * y_train)
    proxF = lambda x,tau: prox(x,tau*lam)
    xinit = As(y_train)

    #run restarted fista
    x_mode,fval,x_adj = rFISTA(proxF, dG, tau, xinit,nIter,mfunc)
    #x_mode,fval, x_adj = ISTA(proxF, dG, tau, xinit,nIter,mfunc)

    return x_mode,fval,x_adj




#setup_base = setup_data(X_train,n,lam,y_train,M,gamma,beta,key)


method_base =  ['PROXL1',
                #'EULA',
                #'EULA_FB',
                'uv_FB',
                #'cart_BOB',
                'cart_PEM',
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
burn_in_base = 10**6
#niter_base = 10**6
#burn_in_base = 10**6
subsamp_base = 1000

#taus = [4/(10*Lf)]*len(method_base)
#taus = [4/(10*Lf), 0.0001, 0.0001, 0.0001]
taus = [0.0001, 0.00001, 0.00001, 0.0001]

#post_prob_neg = np.zeros((n_methods,len(lamvals),len(betavals)))
#post_prob_pos` = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_uncert = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_loss = np.zeros((n_methods,len(lamvals),len(betavals)))



best_runout = np.zeros((len(method_base),len(betavals),niter_base // subsamp_base,n))
best_lam = np.zeros((len(method_base),len(betavals)))
best_mmse = 1e7*np.ones((len(method_base),len(betavals)))
best_mse = 1e7
   
for la in range(len(lamvals)):
    lam = lamvals[la]
    
    "--- REGENERATE MAP ---"
    x_mode,fval,x_adj = obtain_MAP(lam)
    map_mse = np.mean((x_mode @ X_test.T - y_test)**2)
    if map_mse <= best_mse:
        best_mse = map_mse
        best_map = x_mode
        best_lam_map = lam
    
    
    for be in range(len(betavals)):
        beta = betavals[be]
        
        "--- GENERATE SUBDIRECTORY --- "
        subdir = f"lambda = {lam:.0f}, beta = {beta:.0f}"
        print(subdir)
        save_sub = os.path.join("SIMULATION","REGRESS","DIABETES","BELAM",subdir)
        os.makedirs(save_sub, exist_ok=True)
        
        
        save_dist_str = "barplot_MAP.pdf"

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(x_mode)), x_mode, tick_label=columns)
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

            

        "--- SAMPLE UNCERTAINTY-RELATED RESULTS --- "
        "Compute Uncertainty from Your Posterior Samples"
        mean_image = np.mean(runout, axis=(1,3))
        std_image = np.std(runout, axis=(1,3))


        "--- MIXING GRAPH AND TRACE PLOTS---"

        filename = "mix_check.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            #method_diff = np.diff(runout[i,:,:,0], axis=0)
            method_norms = np.linalg.norm(runout[i,:,:,0] - x_mode, axis=1)
            plt.loglog(method_norms, label = method_base[i])
            plt.xlabel("Burn-in")  # Label for x-axis
            plt.ylabel("Error")  # Label for y-axis
            plt.legend()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        
        
        for m in range(n_methods):
            fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharex=True)
            axs = axs.flatten()
    
            for i in range(n):
                axs[i].plot(runout[m, :, i, 0], color='tab:blue')
                axs[i].set_title(columns[i])
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Value')

            fig.suptitle(f'Trace Plots - {method_base[m]}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Adjust for suptitle
            plt.show()
        
        
        #Compute Geweke's convergence criterion
        geweke_pass = np.zeros((n_methods, n), dtype=bool)
        for method in range(n_methods):
                samples = runout[method, :, :, 0]  # shape: (n_samples, n_features)
                for i in range(n):
                    z = geweke_z(samples[:, i])  # Returns array of (index, z-score) pairs
                    geweke_pass[method, i] = np.abs(z) < 2.0  # 95% confidence
        
        for method in range(n_methods):
            failed_count = np.sum(~geweke_pass[method])
            print(f"Method {method_base[method]}: {failed_count} features failed Geweke's test")
        
            
        fig, axes = plt.subplots(3, n_methods, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)

        filename = "weight_many_bars.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            
            # Mean weights
            axes[0, i].bar(range(len(mean_image[i])), mean_image[i], tick_label=columns)
            axes[0, i].set_title(f"{method_base[i]} - Mean Weights")
            
            # Mean vs MAP
            abs_diff = np.abs(mean_image[i] - x_mode)  # Absolute difference per feature
            axes[1, i].bar(range(len(abs_diff)), abs_diff, tick_label=columns)
            axes[1, i].set_title(f"{method_base[i]} - Mean vs MAP")
            
            # Quantile differences
            # Compute log-difference
            quantile_95_method = np.percentile(runout[i, :, :, 0], 95, axis=0)
            quantile_5_method = np.percentile(runout[i, :, :, 0], 5, axis=0)
            quantile_diff_method = np.log(np.abs(quantile_95_method - quantile_5_method))
            
            axes[2, i].bar(range(len(quantile_diff_method)), quantile_diff_method, tick_label=columns)
            axes[2, i].set_title(f"{method_base[i]} - Log-Quantile diffs")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
            
        
        
        save_dist_str = "weight_means.pdf"
        fig_path_bar = os.path.join(save_sub, save_dist_str)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        for i in range(n_methods):
            ttl_str = method_base[i] + ' - Mean Weights per Feature'
        
            axes[i].bar(range(len(mean_image[i])), mean_image[i], tick_label=columns)
            axes[i].set_xticklabels(columns, rotation=45, ha="right",rotation_mode="anchor")
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
        
            axes[i].bar(range(len(abs_diff)), abs_diff, tick_label=columns)
            axes[i].set_xticklabels(columns, rotation=45, ha="right",rotation_mode="anchor")
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
        
            axes[i].bar(range(len(quantile_diff_method)), quantile_diff_method, tick_label=columns)
            axes[i].set_xticklabels(columns, rotation=45, ha="right",rotation_mode="anchor")
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
            
        loss_MAP = mean_squared_error(X_test@x_mode, y_test)
        #our_loss_MAP = helpr.objfun(x_mode,X_test_fun,y_test.reshape(-1))

        
        output_file = os.path.join(save_sub, "results.txt")
        with open(output_file, "w") as f:
            print(f"Results for MAP")
            print(f"Deterministic MSE: {loss_MAP:.4f}")
            f.write(f"Results for MAP\n")
            f.write(f"Deterministic MSE: {loss_MAP:.4f}\n\n")
            for i in range(len(method_base)):
                mean_norm_diff = np.linalg.norm(mean_image[i] - x_mode)
                # Evaluate performance
                fnloss = mean_squared_error(X_test@mean_image[i], y_test)
                mesh_loss[i,la,be] = fnloss
                #our_loss = helpr.objfun(np.mean(runout[i,:,:,:], axis = (0,2)).reshape(-1,1),X_test_fun,y_test)[0]
                tst_wgt = runout[i,:,:,0] @ X_test.T
                median_tst = np.median(tst_wgt,axis=0)
                mmse = np.mean((median_tst - y_test)**2)
                if mmse <= best_mmse[i,be]:
                    best_mmse[i,be] = mmse
                    best_runout[i,be,:,:] = runout[i,:,:,0]
                    best_lam[i,be] = lam
                
                
                print(f"Results for {method_base[i]}")
                print(f"Error vs MAP for {method_base[i]}: {mean_norm_diff}")
                print(f"Test MMSE: {mmse:.4f}")
                print(f"Test Sklearn Loss: {fnloss:.4f}")
                f.write(f"Results for {method_base[i]}\n")
                f.write(f"Error vs MAP for {method_base[i]}: {mean_norm_diff:.6f}\n")
                f.write(f"Test MMSE: {mmse:.4f}\n")
                f.write(f"Test Sklearn Loss: {fnloss:.4f}\n\n")


# Inputs
method_names = method_base  # customize
linestyles = ['-', '--', ':', '-.']  # Different line styles for each method
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
offset = 0.2  # vertical spacing between algorithm lines per feature
markers = ['o', 's', 'D', '^']
extra_offset = 0.1  # for plotting extra spot
normalized_samples = best_runout / np.linalg.norm(best_runout, axis=-1, keepdims=True)
normalized_map = best_map / np.linalg.norm(best_map)

# Create a handle for the "Deterministic" label
deterministic_handle = plt.Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label="LASSO (rFISTA)")
total_methods = n_methods + 1  # Add one extra slot for the deterministic solution

for be in range(len(betavals)):
    ttl_str = 'Posterior mean estimate with credibility intervals'
    # Compute log-difference
    fig, ax = plt.subplots(figsize=(10, 6))
    y_locs = np.arange(n)
    
    for method in range(len(method_base)):
        # Get samples: shape (300, 10)
        samples = normalized_samples[method, be, :, :]
        #samples = best_runout[method, be, :, :]
        means = np.mean(samples, axis=0)  # shape: (10,)
        ci = np.percentile(samples, [2.5, 97.5], axis=0)  # 95% CI
        #ci = np.percentile(samples, [0.5, 99.5], axis=0)  # 99% CI
        print(ci[1,:]-ci[0,:])

        for i in range(n):
            if np.all(ci[:, i] == ci[0, i]):  # If CI is collapsed
                print(f"Warning: Narrow CI for method {method} and feature {i}")
            ypos = y_locs[i] + (method - total_methods / 2) * offset + offset / 2  # stagger vertically
            ax.plot([ci[0, i], ci[1, i]], [ypos, ypos],
                    linestyle=linestyles[method], color=colors[method], label=None)
            ax.plot(means[i], ypos, marker=markers[method], color=colors[method], linestyle='None')

            # Plot extra spot for "Deterministic" without label (we handle that separately)
            #ax.plot(normalized_map[i], i + extra_offset, marker='x', color='black',
            #        linestyle='None', markersize=8)


    # Plot the normalized_map point **once per feature**
    for i in range(n):
        ypos_map = y_locs[i] + (n_methods - total_methods / 2) * offset + offset / 2
        ax.plot(normalized_map[i], ypos_map, marker='x', color='black', linestyle='None', markersize=8, label="LASSO (rFISTA)")


    # Feature labels
    ax.set_yticks(y_locs)
    ax.set_yticklabels(columns)

    # Vertical line at x = 0
    ax.axvline(0.0, linestyle='dotted', color='black')

    # X-axis range
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Normalized coefficient value")
    ax.set_title(f'Posterior mean estimates with 95% CI (β = {betavals[be]})')

    # Legend (only once per method)
    handles = [plt.Line2D([0], [0], color=colors[i], linestyle=linestyles[i], label=method_names[i]) for i in range(n_methods)]
    # Add the deterministic handle separately
    handles.append(deterministic_handle)

    ax.legend(handles=handles, title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.show()









'''MESH PLOTTING'''

# Create meshgrid
Beta, Lambda = np.meshgrid(betavals, lamvals, indexing='ij')



fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "uncertainty_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_uncert[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Predictive Uncertainty")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"Predictive Uncertainty for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "loss_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_loss[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Log-Loss")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"Log-Loss for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()