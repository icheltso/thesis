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
from tensorflow.keras.datasets import mnist

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

save_xtra = os.path.join("SIMULATION","BINARY","MNIST")
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

reload_var = False
train_size = 10000
test_size = 2000

chosen_digit = 7

if reload_var == True or not all(var in globals() for var in ["X_train", "X_test", "y_train", "y_test"]):
    # Load MNIST data
    #X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    #y = y.astype(int)  # Ensure labels are integers

    # Simplify to binary classification (e.g., "digit 0" vs. "not digit 0") - need for our logistic loss term
    #y = (y == 0).astype(int).reshape(-1, 1)
    # Change labels to {-1,1} from {0,1} to fit
    #y = 2 * y - 1

    # Standardize features
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # Split into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)
    
    file_path = os.path.join(save_xtra,"mnist_data.npz")
    
    
    if os.path.exists(file_path):
        # Load from local file
        data = np.load(file_path)
        X_train_full, y_train_full = data["X_train"], data["y_train"]
        X_test_full, y_test_full = data["X_test"], data["y_test"]
        print("MNIST dataset loaded from local file.")
    
    else:
        # Load MNIST dataset
        (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
    
        np.savez_compressed(file_path,
                        X_train=X_train_full, y_train=y_train_full,
                        X_test=X_test_full, y_test=y_test_full)
        print("MNIST dataset downloaded and saved locally.")

    # Flatten the images from 28x28 to 784
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
    X_test_full = X_test_full.reshape(X_test_full.shape[0], -1)

    # Concatenate train and test datasets for preprocessing
    X_full = np.concatenate((X_train_full, X_test_full), axis=0)
    y_full = np.concatenate((y_train_full, y_test_full), axis=0)

    # Simplify to binary classification: "digit 0" vs. "not digit 0"
    y_full_binary = (y_full == chosen_digit).astype(int).reshape(-1, 1)

    # Change labels to {-1, 1} from {0, 1} for the logistic loss
    y_full_binary = 2 * y_full_binary - 1

    # Standardize features
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
    X_full_scaled, y_full_binary, train_size=train_size, test_size=test_size, random_state=42
)







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
def obtain_MAP():
    print("Starting MAP computation")
    Af = lambda x: X_train @ x
    As = lambda x: X_train.T @ x
    prox = lambda x, tau: np.maximum(np.abs(x)-tau, 0)*np.sign(x)
    #mfunc = lambda x: np.sum(np.log(1 + np.exp(-y_train * Af(x)))) + lam*np.linalg.norm(x,ord=1)
    mfunc = lambda x: np.linalg.norm(x,ord=1)
    #tau = 1/20 #stepsize
    #tau = 1/ 50
    #tau = tau_prox
    tau = 4 /  Lf
    nIter =10000
    #dG = lambda x: -(As(y_train * sigmoid(y_train * Af(x))))
    dG = lambda x: As(-sigmoid(-y_train * Af(x)) * y_train)
    proxF = lambda x,tau: prox(x,tau*lam)
    xinit = As(y_train)

    #run restarted fista
    x_mode,fval,x_adj = rFISTA(proxF, dG, tau, xinit,nIter,mfunc)
    #x_mode,fval, x_adj = ISTA(proxF, dG, tau, xinit,nIter,mfunc)

    return x_mode,fval,x_adj

x_mode,fval,x_adj = obtain_MAP()

def clamp(im,vmin=0,vmax=1):
    return np.minimum(np.maximum(im,vmin),vmax)

#plt.imshow(clamp(x_mode).reshape(28,28), cmap='hot',vmin=0,vmax=1)
#plt.imshow(x_mode.reshape(28,28), cmap='hot')
#plt.colorbar(label='Mode')
#plt.show()

#x_mode = x_mode / np.linalg.norm(x_mode)


save_dist_str = "MAP_heatmap.pdf"

# Plot the heatmap
ttl_str_hm = 'Heatmap of MAP'
plt.figure(figsize=(6, 6))
plt.imshow(x_mode.reshape(28,28), cmap='hot', interpolation='nearest')
plt.colorbar(label='Mean Value')
plt.title(ttl_str_hm)
plt.axis('off')
fig_path_bar = os.path.join(save_xtra,save_dist_str)
plt.savefig(fig_path_bar, format="pdf",bbox_inches="tight")
plt.show()


# Compute logits for this batch
logits_MAP = X_test @ x_mode.T  # Shape: (n_test, 1)

# Apply sigmoid to compute probabilities
predictions_MAP = 1 / (1 + np.exp(-logits_MAP))  # Shape: (n_test, 1)

binary_MAP = (predictions_MAP >= 0.5).astype(int)


setup_base = setup_data(X_train,n,lam,y_train,M,gamma,beta,key)


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
burn_in_base = 500
niter_base = 500
subsamp_base = 100

#niter_base = 3*10**5
#burn_in_base = 10**4
#subsamp_base = 100



#taus = [tau_prox,0.001]
taus = [4/Lf,0.001,0.001,0.0005]
runout, burn_time, sample_time = Runner(setup_base, method_base).runner(niter_base, taus, burn_in_base, subsamp_base) 


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
fig_path_bar = os.path.join(save_xtra,save_dist_str)
plt.savefig(fig_path_bar, format="pdf",bbox_inches="tight")
plt.show()




"--- PLOT TIMES"

sum_burn_time = np.sum(burn_time, axis = 1)
sum_samp_time = np.sum(sample_time, axis = 1)

"Create a bar-chart of mean per-iteration times"
filename_bar = "times_bar.pdf"
fig_path4 = os.path.join(save_xtra,filename_bar)
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

n_methods = len(method_base)
n_samp = runout[0,:,0,0].shape[0]
n_test = X_test.shape[0]
#predictions = np.zeros((len(method_base),n_samp, n_test))

batch_size = 1000  # Number of samples to process at a time

# Initialize storage for averaged predictions
mean_predictions = np.zeros((n_methods, n_test))  # Shape: (n_methods, n_test)

# Loop over methods
for i in range(n_methods):
    method_predictions = np.zeros((n_samp, n_test))  # Temporary storage for this method's predictions

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
        method_predictions[start:end, :] = predictions_batch.T

    # Average predictions over all samples
    mean_predictions[i] = np.mean(method_predictions, axis=0)

# Convert probabilities to binary predictions: threshold at 0.5
binary_predictions = (mean_predictions >= 0.5).astype(int)

# Adjust y_test from {-1, +1} to {0, 1} for compatibility between loss function and metrics
y_test_adjusted = (y_test == 1).astype(int)


    
    
mean_image = np.mean(runout, axis=(1,3)).reshape(n_methods, 28, 28)

for i in range(n_methods):
    #method_diff = np.diff(runout[i,:,:,0], axis=0)
    method_norms = np.linalg.norm(runout[i,:,:,0] - x_mode, axis=1)
    plt.semilogy(method_norms)


for i in range(n_methods):
    
    save_dist_str = "heatmap_" + method_base[i] + ".pdf"
    fig_path_bar = os.path.join(save_xtra,save_dist_str)
    
    
    # Plot the heatmap
    ttl_str_hm = method_base[i] + ' - Heatmap of Mean Weights'
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_image[i], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Mean Value')
    plt.title(ttl_str_hm)
    plt.axis('off')
    plt.savefig(fig_path_bar, format="pdf",bbox_inches="tight")
    plt.show()
    
    save_dist_str = "heatmap_vs_MAP_" + method_base[i] + ".pdf"
    fig_path_bar2 = os.path.join(save_xtra,save_dist_str)
    
    # Plot the heatmap difference against MAP
    ttl_str_hmd = method_base[i] + ' vs MAP abs. difference - Heatmap'
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(mean_image[i] - x_mode.reshape(28,28)), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Mean Value')
    plt.title(ttl_str_hmd)
    plt.axis('off')
    plt.savefig(fig_path_bar2, format="pdf",bbox_inches="tight")
    plt.show()
    
    # Calculate the 95th and 5th quantiles for each pixel 
    quantile_95_method = np.percentile(runout[i,:,:,0], 95, axis=0).reshape(28, 28)
    quantile_5_method = np.percentile(runout[i,:,:,0], 5, axis=0).reshape(28, 28)
    # Calculate the difference between the 95th and 5th quantile
    quantile_diff_method = np.log(np.abs(quantile_95_method - quantile_5_method))
    
    save_dist_str = "quantile_" + method_base[i] + ".pdf"
    fig_path_bar3 = os.path.join(save_xtra,save_dist_str)

    # Plot the heatmap of the difference
    ttl_str_hm_quant = method_base[i] + ' - Difference between 95th and 5th Quantiles'
    plt.figure(figsize=(6, 6))
    plt.imshow(quantile_diff_method, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Log-Quantile Difference log|95th - 5th|')
    plt.title(ttl_str_hm_quant)
    plt.axis('off')
    plt.savefig(fig_path_bar3, format="pdf",bbox_inches="tight")
    plt.show()
    
    
helpr = Helper(setup_base)
X_test_fun = lambda x: X_test @ x
#X_test_s_fun = lambda x: X_test.T @ x

#ytst_M = jnp.tile( y_test[:,None],(1,M))
    
accuracy_MAP = accuracy_score(y_test_adjusted, binary_MAP)
roc_auc_MAP = roc_auc_score(y_test_adjusted, predictions_MAP)
logloss_MAP = log_loss(y_test_adjusted, predictions_MAP)
our_logloss_MAP = helpr.objfun(x_mode,X_test_fun,y_test.reshape(-1))
print(f"Results for MAP")
print(f"Test Accuracy: {accuracy_MAP:.4f}")
print(f"Test ROC-AUC: {roc_auc_MAP:.4f}")
print(f"Test Log-Loss: {our_logloss_MAP:.4f}")
print(f"Test Sklearn Log-Loss: {logloss_MAP:.4f}")
for i in range(len(method_base)):
    method_norms = np.linalg.norm(runout[i,:,:,0] - x_mode, axis=1)
    # Evaluate performance
    accuracy = accuracy_score(y_test_adjusted, binary_predictions[i])
    roc_auc = roc_auc_score(y_test_adjusted, mean_predictions[i])
    logloss = log_loss(y_test_adjusted, mean_predictions[i])
    our_logloss = helpr.objfun(runout[i,-1,:,:],X_test_fun,y_test)[0]
    print(f"Results for {method_base[i]}")
    print(f"Error vs MAP for {method_base[i]}: {method_norms[-1]}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test Log-Loss: {our_logloss:.4f}")
    print(f"Test Sklearn Log-Loss: {logloss:.4f}")
    
    
    


    
    
