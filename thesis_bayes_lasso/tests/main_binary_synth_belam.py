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


# Set parameters
reload_var = False
#train_size = 10000
#test_size = 2000

# File path to save the dataset locally
file_path = os.path.join(save_xtra, "synth_data.npz")


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
    noise = np.random.normal(0, noisy_var, size=(n, noise_dim))  # Gaussian noise
    X_full = np.hstack((X, noise))  # Combine original data with noise
    
    Y = 2 * Y - 1
    
    return X_full,Y

n = 200 #number of data points
#noise_d = 28

bounds = np.array([[-2,3],[-4,5],[-1,2]])
noise_d = 97
noisy_var = 1
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


feature_names = []

for i in range(dim):
    strname = 'X' + str(i + 1)
    feature_names.append(strname)
    
for i in range(noise_d):
    strname = 'N' + str(i + 1)
    feature_names.append(strname)


print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


"--- DEFINE BETA AND LAMBDA RANGES HERE ---"


#betavals = np.logspace(0,np.log10(50),3)
betavals = [0.1,1,10]
maxlam = jnp.max(jnp.abs(X_train.T@y_train))
lamvals = np.logspace(-1,np.log10(maxlam),20)
#lamvals = np.linspace(1,maxlam,6)

#betavals= [1, 100]
#lamvals = [1,100]



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
beta = 3000
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
            prob_pred = X_val_fold @ x_mode  # Predicted log-odds
            #y_pred_class = (prob_pred > 0.5).astype(np.int32)  # Convert to class labels using 0.5 threshold
            y_pred_class = np.sign(prob_pred)

            #accuracy = accuracy_score(y_val_fold, y_pred_class)
            accuracy = binary_accuracy(y_val_fold, y_pred_class)
            fold_accuracies.append(accuracy)

        mean_cv_accuracy = np.mean(fold_accuracies)
        mean_accuracies.append(mean_cv_accuracy)
        print(f"Lambda = {lam:.4f}, CV Accuracy = {mean_cv_accuracy:.4f}")

    print(mean_accuracies)

    best_idx = np.argmax(mean_accuracies)
    best_lambda = lam_vals[best_idx]
    print(f"\nBest lambda = {best_lambda:.4f} with average CV Accuracy = {mean_accuracies[best_idx]:.4f}")
    return best_lambda, mean_accuracies

# Example usage: tuning lambda over a grid
best_lam, accuracy_vals = lasso_cv_custom(X_train, y_train, lamvals)

# Final evaluation on the test set using best lambda
x_mode, _, _ = obtain_MAP(X_train, y_train, best_lam)
prob_pred_test = X_test @ x_mode
logits_tst = 1 / (1 + np.exp(-prob_pred_test))
y_test_pred = (prob_pred_test > 0.5).astype(np.int32)  # Convert to class labels
#y_test_pred = np.sign(prob_pred_test)
#test_accuracy = accuracy_score(y_test_adjusted, y_test_pred)
test_accuracy = binary_accuracy(y_test_adjusted, y_test_pred)

print(f"Test Accuracy using MAP with λ={best_lam}: {test_accuracy:.4f}")



def mean_predictive_entropy(pred_probs):
    """
    Compute mean predictive entropy for given predictive probabilities.

    Parameters:
    - pred_probs: (S, N) array, where pred_probs[s, i] is the probability for test point i, sample s.

    Returns:
    - Mean predictive entropy (scalar).
    """
    mean_probs = np.mean(pred_probs, axis=0)  # Average over S samples (posterior samples)
    entropy = -mean_probs * np.log(mean_probs) - (1 - mean_probs) * np.log(1 - mean_probs)
    return np.mean(entropy)  # Average over test cases






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

niter_base = 10**4
burn_in_base = 10**4
subsamp_base = 1

#niter_base = 3*10**5
#burn_in_base = 10**5
#subsamp_base = 1000

#taus = [4/Lf]*len(method_base)
taus = [4/Lf, 1/(2*Lf), 1/(2*Lf), 1/(2*Lf)]

#post_prob_neg = np.zeros((n_methods,len(lamvals),len(betavals)))
#post_prob_pos` = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_uncert = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_accuracy = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_logloss = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_rocauc = np.zeros((n_methods,len(lamvals),len(betavals)))
pred_ent = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_pred_pos = np.zeros((n_methods,len(lamvals),len(betavals)))
mesh_pred_neg = np.zeros((n_methods,len(lamvals),len(betavals)))

mesh_mce = np.zeros((n_methods,len(lamvals),len(betavals)))

    
for la in range(len(lamvals)):
    lam = lamvals[la]
    
    "--- REGENERATE MAP ---"
    x_mode,fval,x_adj = obtain_MAP(X_train, y_train, lam)

    # Compute logits for this batch
    logits_MAP = X_test @ x_mode.T  # Shape: (n_test, 1)

    # Apply sigmoid to compute probabilities
    predictions_MAP = 1 / (1 + np.exp(-logits_MAP))  # Shape: (n_test, 1)

    binary_MAP = (predictions_MAP >= 0.5).astype(int)
    
    
    
    
    for be in range(len(betavals)):
        beta = betavals[be]
        
        "--- GENERATE SUBDIRECTORY --- "
        subdir = f"lambda = {lam:.0f}, beta = {beta:.0f}"
        print(subdir)
        save_sub = os.path.join("SIMULATION","BINARY","SYNTH","BELAM",subdir)
        os.makedirs(save_sub, exist_ok=True)
        
        
        save_dist_str = "barplot_MAP.pdf"

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(x_mode)), x_mode, tick_label=feature_names)
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






        #Firstly, check that chain is well-mixed at start of iterations

        
        n_samp = runout[0,:,0,0].shape[0]
        n_test = X_test.shape[0]
        #predictions = np.zeros((len(method_base),n_samp, n_test))

        batch_size = 1000  # Number of samples to process at a time

        # Initialize storage for averaged predictions
        mean_predictions = np.zeros((n_methods, n_test))  # Shape: (n_methods, n_test)

        method_predictions = np.zeros((n_methods, n_samp, n_test))

        # Loop over methods
        for i in range(n_methods):
            #method_predictions = np.zeros((n_samp, n_test))  # Temporary storage for this method's predictions

            # Process weights in chunks
            for start in range(0, n_samp, batch_size):
                end = min(start + batch_size, n_samp)
                print(f"Processing samples {start} to {end} for method {method_base[i]}...")

                # Extract a batch of weights
                weights_batch = runout[i, start:end, :, 0]  # Shape: (batch_size, n_features)

                # Compute logits for this batch
                logits_batch = X_test @ weights_batch.T  # Shape: (n_test, batch_size)

                # Apply sigmoid to compute probabilities
                predictions_batch = 1 / (1 + np.exp(-logits_batch))  # Shape: (n_test, batch_size)

                # Store the predictions (transpose to align shapes)
                method_predictions[i,start:end, :] = predictions_batch.T

            # Average predictions over all samples
            mean_predictions[i] = np.mean(method_predictions[i], axis=0)

        # Convert probabilities to binary predictions: threshold at 0.5
        binary_predictions = (mean_predictions >= 0.5).astype(int)


        
        
        



        mce_list = []  # To store mean classification errors for each method
        
        # Loop over methods
        for i in range(n_methods):
            samples = runout[i]
            mce_method_list = []  # Store mean classification error for each posterior sample set
        
            # Process weights in chunks
            for s in range(n_samp):  # For each subsampled iteration
                mce_particles = []  # Store the mean classification error for each particle
                
                for p in range(M):  # For each posterior sample from the method
                    beta = samples[s, :, p]  # shape: (x_dim,)
                    
                    # Compute logits: X_test @ beta (this gives log-odds)
                    logits = X_test @ beta  # shape: (n_test,)
                    
                    # Compute predicted probabilities using sigmoid function
                    prob = 1 / (1 + jnp.exp(-logits))  # shape: (n_test,)
                    
                    # Classify based on 0.5 threshold
                    y_pred_class = (prob > 0.5).astype(jnp.float32)  # shape: (n_test,)
                    
                    # Calculate classification error (misclassification rate)
                    classification_error = jnp.mean(y_pred_class != y_test)  # shape: scalar, classification error for this posterior sample
                    
                    mce_particles.append(classification_error)
                
                # Compute the mean classification error for this subsample s
                mce_method_list.append(jnp.mean(jnp.array(mce_particles)))  # shape: scalar, mean classification error for this sample
        
            # Compute the median of mean classification errors (MCE) across all subsampled iterations (s = 1...n_samp)
            median_mce = jnp.median(jnp.array(mce_method_list))  # shape: scalar, median classification error for this method
            mce_list.append(median_mce)  # Store for this method
        
        # Final MCE array for all algorithms: shape (no_algs,)
        mesh_mce[:, la, be] = np.array(mce_list)  # Store for plotting or analysis






        "--- CLASSIFIER UNCERTAINTY-RELATED RESULTS ---"
        mean_true_ALL = []
        mean_false_ALL = []

        "Compute mean prediction for true/false cases for each method"
        for i in range(n_methods):
            mean_true_ALL.append(mean_predictions[i][y_test == 1])
            mean_false_ALL.append(mean_predictions[i][y_test == -1])
            pred_ent[i,la,be] = mean_predictive_entropy(method_predictions[i])
            
            
        "--- COMPUTE MESH STUFF ---"
        var_predictions = np.var(method_predictions, axis = 1)
        mesh_uncert[:,la,be] = np.var(method_predictions, axis = (1,2))
        #post_prob_pos[:,la,be] = np.mean()
        #post_prob_neg[:,la,be]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()

        "Plot predictive mean vs predicitive variance"
        filename = "predictive_mean_vs_var.pdf"
        fig_path = os.path.join(save_sub,filename)
        ylimit = np.max(var_predictions)
        for i in range(n_methods):
            ttl_str = method_base[i] + " - Uncertainty Across Test Cases"
            axes[i].scatter(mean_predictions[i], var_predictions[i])
            axes[i].set_ylim(0,ylimit)
            axes[i].set_xlabel("Mean Posterior Probability", fontsize=12)
            axes[i].set_ylabel("Variance of Posterior Probability", fontsize=12)
            axes[i].set_title(ttl_str, fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
        
        "Plot same as above but on one figure with different levels of opacity"
        ttl_str = "Uncertainty Across Test Cases"
        filename = "predictive_mean_vs_var_single.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            plt.scatter(mean_predictions[i], var_predictions[i], label = method_base[i], alpha = 0.5)
        plt.ylim(0,ylimit)
        plt.xlabel("Mean Posterior Probability", fontsize=12)
        plt.ylabel("Variance of Posterior Probability", fontsize=12)
        plt.title(ttl_str, fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()


        "Histograms for mean posterior probabilities over test cases"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        filename = "predictive_mean_hist.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            ttl_str = method_base[i] + " - Distribution of Mean Posterior Probabilities Across Test Cases"
            axes[i].hist(mean_predictions[i], bins=20, alpha=0.7)
            axes[i].set_xlabel("Mean Posterior Probability")
            axes[i].set_ylabel("Number of Test Cases")
            axes[i].set_title(ttl_str)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
        
        
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        filename = "predictive_mean_hist_class.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            ttl_str = method_base[i] + " - Distribution of Mean Posterior Test Probabilities"
            axes[i].hist(mean_true_ALL[i], bins=20, alpha=0.5, label="Cancer Positive")
            axes[i].hist(mean_false_ALL[i], bins=20, alpha=0.5, label="Cancer Negative")
            axes[i].set_xlabel("Mean Posterior Probability")
            axes[i].set_ylabel("Number of Test Cases")
            axes[i].set_title(ttl_str)
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
        
            
        "Compute percentiles for predictions"
        percentile_probs_low = np.percentile(method_predictions, 2.5, axis=1)
        percentile_probs_hi = np.percentile(method_predictions, 97.5, axis=1)
        uncertainty_range = percentile_probs_hi - percentile_probs_low
        correct_uncertainty = []
        incorrect_uncertainty = []
        for i in range(n_methods):
            correct_uncertainty.append(uncertainty_range[i][y_test_adjusted == binary_predictions[i]])  # Uncertainty for correct cases
            incorrect_uncertainty.append(uncertainty_range[i][y_test_adjusted != binary_predictions[i]])  # Uncertainty for incorrect cases

        "Plot histograms for uncertainty in predictions"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        filename = "predictive_uncert_hist.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods): 
            axes[i].hist(correct_uncertainty[i], alpha=0.5, label="Correctly Classified", bins=20)
            axes[i].hist(incorrect_uncertainty[i], alpha=0.5, label="Incorrectly Classified", bins=20)
            axes[i].set_xlabel("95% CI Width (Uncertainty)")
            axes[i].set_ylabel("Number of Test Cases")
            axes[i].set_title("Uncertainty in Correct vs Incorrect Classifications")
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()

        "percentile plots"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        filename = "predictive_posterior_scatter.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            axes[i].scatter(range(len(y_test)), mean_predictions[i], label="Mean Posterior Probability", alpha=0.6)
            axes[i].fill_between(range(len(y_test)), percentile_probs_low[i], percentile_probs_hi[i], color='b', alpha=0.3, label="95% CI")
            axes[i].scatter(range(len(y_test)), y_test_adjusted, color='red', marker='x', label="True Labels")
            axes[i].set_xlabel("Test Case Index")
            axes[i].set_ylabel("Posterior Probability")
            axes[i].set_title("Mean Posterior Probability with 95% Credible Interval")
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
            
        "kde plots for percentiles"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        filename = "predictive_posterior_kde.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            sns.kdeplot(mean_predictions[i], ax=axes[i], label="Mean Posterior", shade=True)
            sns.kdeplot(percentile_probs_low[i], ax=axes[i], label="2.5% Percentile", linestyle="--")
            sns.kdeplot(percentile_probs_hi[i], ax=axes[i], label="97.5% Percentile", linestyle="--")
            axes[i].set_xlabel("Posterior Probability")
            axes[i].set_title(f"{method_base[i]} - Density of Posterior Probabilities")
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()


        "--- SAMPLE UNCERTAINTY-RELATED RESULTS --- "
        feature_std = np.std(X_train, axis=0)
        "Compute Uncertainty from Your Posterior Samples"
        mean_image = np.mean(runout, axis=(1,3))
        std_image = np.std(runout, axis=(1,3))
        
        filename = "weight_data_vs_post_bar.pdf"
        fig_path = os.path.join(save_sub,filename)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
        axes = axes.flatten()
        for i in range(n_methods):
            axes[i].bar(feature_names, feature_std, alpha=0.5, label="Data Std")
            axes[i].bar(feature_names, std_image[i], alpha=0.5, label="Posterior Std")
            axes[i].set_xticklabels(feature_names, rotation=45, ha="right",rotation_mode="anchor")
            axes[i].set_xlabel("Feature Index")
            axes[i].set_ylabel("Standard Deviation")
            axes[i].legend()
            axes[i].set_title(f"{method_base[i]} - Data vs. Posterior Uncertainty")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()


        "--- MIXING GRAPH ---"

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
            
        fig, axes = plt.subplots(3, n_methods, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)

        filename = "weight_many_bars.pdf"
        fig_path = os.path.join(save_sub,filename)
        for i in range(n_methods):
            
            # Mean weights
            axes[0, i].bar(range(len(mean_image[i])), mean_image[i], tick_label=feature_names)
            axes[0, i].set_title(f"{method_base[i]} - Mean Weights")
            
            # Mean vs MAP
            abs_diff = np.abs(mean_image[i] - x_mode)  # Absolute difference per feature
            axes[1, i].bar(range(len(abs_diff)), abs_diff, tick_label=feature_names)
            axes[1, i].set_title(f"{method_base[i]} - Mean vs MAP")
            
            # Quantile differences
            # Compute log-difference
            quantile_95_method = np.percentile(runout[i, :, :, 0], 95, axis=0)
            quantile_5_method = np.percentile(runout[i, :, :, 0], 5, axis=0)
            quantile_diff_method = np.log(np.abs(quantile_95_method - quantile_5_method))
            
            axes[2, i].bar(range(len(quantile_diff_method)), quantile_diff_method, tick_label=feature_names)
            axes[2, i].set_title(f"{method_base[i]} - Log-Quantile diffs")
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
            
        accuracy_MAP = accuracy_score(y_test_adjusted, binary_MAP)
        roc_auc_MAP = roc_auc_score(y_test_adjusted, predictions_MAP)
        logloss_MAP = log_loss(y_test_adjusted, predictions_MAP)
        our_logloss_MAP = helpr.objfun(x_mode,X_test_fun,y_test.reshape(-1))
        
        mesh_accuracy[:,]
        
        output_file = os.path.join(save_sub, "results.txt")
        with open(output_file, "w") as f:
            print(f"Results for MAP")
            print(f"Test Accuracy: {accuracy_MAP:.4f}")
            print(f"Test ROC-AUC: {roc_auc_MAP:.4f}")
            print(f"Test Log-Loss: {our_logloss_MAP:.4f}")
            print(f"Test Sklearn Log-Loss: {logloss_MAP:.4f}")
            f.write(f"Results for MAP\n")
            f.write(f"Test Accuracy: {accuracy_MAP:.4f}\n")
            f.write(f"Test ROC-AUC: {roc_auc_MAP:.4f}\n")
            f.write(f"Test Log-Loss: {our_logloss_MAP:.4f}\n")
            f.write(f"Test Sklearn Log-Loss: {logloss_MAP:.4f}\n\n")
            for i in range(len(method_base)):
                mean_norm_diff = np.linalg.norm(mean_image[i] - x_mode)
                # Evaluate performance
                accuracy = accuracy_score(y_test_adjusted, binary_predictions[i])
                mesh_accuracy[i,la,be] = accuracy
                roc_auc = roc_auc_score(y_test_adjusted, mean_predictions[i])
                mesh_rocauc[i,la,be] = roc_auc
                logloss = log_loss(y_test_adjusted, mean_predictions[i])
                mesh_logloss[i,la,be] = logloss
                our_logloss = helpr.objfun(np.mean(runout[i,:,:,:], axis = (0,2)).reshape(-1,1),X_test_fun,y_test)[0]
                print(f"Results for {method_base[i]}")
                print(f"Error vs MAP for {method_base[i]}: {mean_norm_diff}")
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Median Mean Classification Error: {mesh_mce[i, la, be]}")
                print(f"Mean Probability for True: {np.mean((mean_true_ALL[i])):.4f}")
                mesh_pred_pos[i,la,be] = np.mean((mean_true_ALL[i]))
                print(f"Mean Probability for False: {np.mean((mean_false_ALL[i])):.4f}")
                mesh_pred_neg[i,la,be] = np.mean((mean_false_ALL[i]))
                print(f"Test ROC-AUC: {roc_auc:.4f}")
                print(f"Test Log-Loss: {our_logloss:.4f}")
                print(f"Test Sklearn Log-Loss: {logloss:.4f}")
                f.write(f"Results for {method_base[i]}\n")
                f.write(f"Error vs MAP for {method_base[i]}: {mean_norm_diff:.6f}\n")
                f.write(f"Test Accuracy: {accuracy:.4f}\n")
                f.write(f"Median Mean Classification Error: {mesh_mce[i, la, be]}\n")
                f.write(f"Mean Probability for True: {np.mean(mean_true_ALL[i]):.4f}\n")
                f.write(f"Mean Probability for False: {np.mean(mean_false_ALL[i]):.4f}\n")
                f.write(f"Test ROC-AUC: {roc_auc:.4f}\n")
                f.write(f"Test Log-Loss: {our_logloss:.4f}\n")
                f.write(f"Test Sklearn Log-Loss: {logloss:.4f}\n\n")



# Prepare arrays to store best values
best_mmce = np.zeros(n_methods)
best_lam = np.zeros(n_methods)
best_beta = np.zeros(n_methods)

for i in range(n_methods):
    # Find index of min MMSE for method i
    min_idx = np.unravel_index(np.argmin(mesh_mce[i]), mesh_mce[i].shape)
    lam_idx, beta_idx = min_idx

    best_mmce[i] = mesh_mce[i, lam_idx, beta_idx]
    best_lam[i] = lamvals[lam_idx]
    best_beta[i] = betavals[beta_idx]

# Optional: print results
print(f"Test MSE using MAP with λ={best_lam}: {test_accuracy:.4f}")
for i in range(n_methods):
    print(f"Method {method_base[i]}: Best MMCE = {best_mmce[i]:.4f} at λ = {best_lam[i]:.4f}, β = {best_beta[i]:.4f}")





# Create meshgrid
Beta, Lambda = np.meshgrid(betavals, lamvals, indexing='ij')



fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "mmce_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    color_map = ax.pcolormesh(Lambda, Beta, mesh_mce[i].T, cmap="coolwarm", shading='auto')
    
    # Colorbar for accuracy
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("MMCE")

    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"{method_base[i]} - Median MCE")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "accuracy_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    color_map = ax.pcolormesh(Lambda, Beta, mesh_accuracy[i].T, cmap="coolwarm", shading='auto')
    
    # Colorbar for accuracy
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Accuracy")

    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"{method_base[i]} - Accuracy")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()



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
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "logloss_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_logloss[i].T, cmap="coolwarm", shading='auto')
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
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "roc_auc_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_rocauc[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("ROC-AUC")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"ROC-AUC for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "pred_ent_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, pred_ent[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Predictive Entropy")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"Predictive Entropy for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "pos_prob_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_pred_pos[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Mean Probability for Positive Class")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"Mean Probability for Positive Class for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "neg_prob_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)
for i in range(n_methods): 
    
    ax = axes[i]
    # Plot the meshgrid heatmap
    color_map = ax.pcolormesh(Lambda, Beta, mesh_pred_neg[i].T, cmap="coolwarm", shading='auto')
    # Add uncertainty as contour lines on top
    # Add colorbar
    cbar = fig.colorbar(color_map, ax=ax)
    cbar.set_label("Mean Probability for Negative Class")
    
    # Labels and title
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_title(f"Mean Probability for Negative Class for {method_base[i]}")
    ax.set_xlim([min(lamvals), max(lamvals)])
    ax.set_ylim([min(betavals), max(betavals)])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "acc_vs_uncert_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)

for i in range(n_methods): 
    ax = axes[i]
    c = ax.contourf(Lambda, Beta, mesh_accuracy[i].T, cmap='coolwarm', levels=20)
    contour_lines = ax.contour(Lambda, Beta, mesh_uncert[i].T, colors='black', linestyles='dashed')
    ax.clabel(contour_lines, fmt="%.2f", inline=True, fontsize=10)  # Label entropy contours
    fig.colorbar(c, ax=ax, label="Accuracy")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    #ax.set_xscale('log')  # Log scale for Beta
    ax.set_yscale('symlog')  # Log scale for Lambda
    ax.set_xscale('symlog')
    ax.set_title(f'{method_base[i]}: Accuracy (Color) + Pred. Uncertainty (Contours)')

plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "acc_vs_ent_mesh.pdf"
fig_path = os.path.join(save_xtra,filename)

for i in range(n_methods): 
    ax = axes[i]
    c = ax.contourf(Lambda, Beta, mesh_accuracy[i].T, cmap='coolwarm', levels=20)
    contour_lines = ax.contour(Lambda, Beta, pred_ent[i].T, colors='black', linestyles='dashed')
    ax.clabel(contour_lines, fmt="%.2f", inline=True, fontsize=10)  # Label entropy contours
    fig.colorbar(c, ax=ax, label="Accuracy")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    #ax.set_xscale('log')  # Log scale for Beta
    ax.set_yscale('symlog')  # Log scale for Lambda
    ax.set_xscale('symlog')
    ax.set_title(f'{method_base[i]}: Accuracy (Color) + Entropy (Contours)')

plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()
    

fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 3 rows (metrics), 4 columns (methods)
axes = axes.flatten()
filename = "acc_vs_roc.pdf"
fig_path = os.path.join(save_xtra,filename)

for i in range(n_methods): 
    ax = axes[i]
    c = ax.contourf(Lambda, Beta, mesh_accuracy[i].T, cmap='coolwarm', levels=20)
    contour_lines = ax.contour(Lambda, Beta, mesh_rocauc[i].T.T, colors='black', linestyles='dashed')
    ax.clabel(contour_lines, fmt="%.2f", inline=True, fontsize=10)  # Label entropy contours
    fig.colorbar(c, ax=ax, label="Accuracy")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel(r"Inverse Temperature $\beta$")
    #ax.set_xscale('log')  # Log scale for Beta
    ax.set_yscale('symlog')  # Log scale for Lambda
    ax.set_xscale('symlog')
    ax.set_title(f'{method_base[i]}: Accuracy (Color) + ROC-AUC (Contours)')

plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()