# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:10:52 2024

@author: ichel
"""

'Create separate file for visualizing some basic results.'
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import Helper
from util import get_N_HexCol, legend_without_duplicate_labels

subfolder_name = "SIMULATION"
current_directory = os.getcwd()
sim_path = os.path.join(current_directory, subfolder_name)

class Visual():
    def __init__(self, setup, alg_id, x_arr, mean_arr, std_arr, tm_avg, title_xtra, save_xtra):
        current_directory = os.getcwd()
        #file_path = os.path.join(current_directory, "data.npz")
        #data = np.load(file_path)
        #file_path = os.path.join(current_directory, "data.pkl")
        #with open(file_path, 'rb') as file:
        #    loaded_dump = dill.load(file)
        datadict = setup.get_data()
        
        #self.lam = data["lam"]
        self.lam = datadict["lam"]
        self.alg_id = alg_id
        self.x_arr = x_arr
        self.mean_arr = mean_arr
        self.std_arr = std_arr
        self.tm_avg = tm_avg
        self.title_xtra = title_xtra
        self.save_xtra = save_xtra
        self.helpr = Helper(setup)
        self.subfolder_path = os.path.join(sim_path, self.save_xtra)
        if not os.path.exists(self.subfolder_path):
            os.makedirs(self.subfolder_path)
    
    'For 1-D Case plot histograms for selected methods'
    'n_bins - number of bins for histograms'
    'no_stds - defines range, in standard deviations, of the target distribution plot.'
    def hist_1d(self, n_bins, no_stds):
        if (np.size(self.x_arr,1) > 1):
            raise ValueError("To plot histogram/kde, dimension must be 1.")
        
        no_methods = len(self.alg_id)
        
        #determine ranges for graph
        meann = 0
        stdd = max(self.std_arr[:,0,-1])
        for i in range(no_methods):
            meann = meann + self.mean_arr[i,0,-1]/no_methods
        
        #Set number of bins for histogram
        #n_bins = 100
        """Obtain target density."""
        z_for_dens = np.linspace(meann-no_stds*stdd,meann+no_stds*stdd,100)
        p_x = self.helpr.get_dens_lsqr(z_for_dens,self.lam)
        ttl_str = 'Distribution comparison. ' + self.title_xtra
        
        for j in range(no_methods):
            fig, ax = plt.subplots()
            ax.plot(z_for_dens, p_x, label = 'target density')
            ax.hist(self.x_arr[j,0,:], bins=n_bins, density = True, label = self.alg_id[j])
            ax.set_title(ttl_str)
            ax.legend()
            filename = '/' + self.alg_id[j] + '.png'
            fig.savefig(self.subfolder_path + filename)
        
            
    'Create KDE plots'
    def kde_1d(self, no_stds):
        if (np.size(self.x_arr,1) > 1):
            raise ValueError("To plot histogram/kde, dimension must be 1.")
        
        no_methods = len(self.alg_id)
        #palette_num = get_N_HexCol(no_methods+1)
        palette_num = plt.get_cmap('tab10')
        #determine ranges for graph
        meann = 0
        stdd = max(self.std_arr[:,0,-1])
        for i in range(no_methods):
            meann = meann + self.mean_arr[i,0,-1]/no_methods
        
        #Set number of bins for histogram
        #n_bins = 100
        """Obtain target density."""
        z_for_dens = np.linspace(meann-no_stds*stdd,meann+no_stds*stdd,100)
        p_x = self.helpr.get_dens_lsqr(z_for_dens,self.lam)
        sns.set()
        fig, ax = plt.subplots()
        ax.plot(z_for_dens, p_x, color = palette_num(0), linestyle = '--', label = 'target density')
        ttl_str = 'Distribution comparison. ' + self.title_xtra
        for i in range(no_methods):
            ax = sns.kdeplot(self.x_arr[i,0,:], color = palette_num(i+1), label = self.alg_id[i])
            
        ax.set_title(ttl_str)
        ax.legend()
        filename = '/kdeplot.png'
        fig.savefig(self.subfolder_path + filename)
        
    'Plot sample means'
    def smpl_mean(self, strt, ends):
        no_methods = len(self.alg_id)
        #print(self.lam)
        xsol = self.helpr.solve_lasso_lsqr(self.lam)
        #print(xsol)
        #n2 = len(xsol)
        niter = np.size(self.mean_arr,1)
        xsol = np.tile(xsol.reshape(-1,1),niter).T
        fig_mean, ax_mean = plt.subplots()
        #palette_num = get_N_HexCol(no_methods+1)
        palette_num = plt.get_cmap('tab10')
        ttl_str = 'Convergence of sample means. ' + self.title_xtra
        burn_in_mean = self.mean_arr[:,strt:ends,:]
        iters_range = np.arange(strt, ends)
        for i in range(no_methods):
            ax_mean.plot(iters_range, burn_in_mean[i],color = palette_num(i), label=self.alg_id[i])
        ax_mean.plot(iters_range, xsol[strt:ends], color = palette_num(no_methods), linestyle = '--', label = 'target mean')
            
        ax_mean.set_title(ttl_str)
        legend_without_duplicate_labels(ax_mean)
        filename = '/sample_means_' + str(strt) + '_' + str(ends) + '_.png'
        fig_mean.savefig(self.subfolder_path + filename)
        
    'Plot Time averages for a single particle - use with numpy methods'
    def time_avg(self, strt, ends):
        no_methods = len(self.alg_id)
        xsol = self.helpr.solve_lasso_lsqr(self.lam)
        niter = np.size(self.mean_arr,1)
        xsol = np.tile(xsol.reshape(-1,1),niter).T
        fig_mean, ax_mean = plt.subplots()
        #palette_num = get_N_HexCol(no_methods+1)
        palette_num = plt.get_cmap('tab10')
        ttl_str = 'Convergence of time averages. ' + self.title_xtra
        burn_in_mean = self.tm_avg[:,strt:ends,:]
        iters_range = np.arange(strt, ends)
        for i in range(no_methods):
            ax_mean.plot(iters_range, burn_in_mean[i],color = palette_num(i), label=self.alg_id[i])
        ax_mean.plot(iters_range, xsol[strt:ends], color = palette_num(no_methods), linestyle = '--', label = 'target mean')
            
        ax_mean.set_title(ttl_str)
        legend_without_duplicate_labels(ax_mean)
        filename = '/time_averages_' + str(strt) + '_' + str(ends) + '_.png'
        fig_mean.savefig(self.subfolder_path + filename)
            
        
            