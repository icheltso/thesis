# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:50:48 2024

@author: ichel
"""

import arviz as az
import numpy as np

# Simulate some MCMC samples (for example purposes)
# Let's assume we have a 2D array where rows are chains and columns are samples
mcmc_samples = np.random.randn(1000, 2)  # 1000 samples for 2 parameters

# Create an InferenceData object
idata = az.from_dict({"posterior": {"param1": mcmc_samples[:, 0], "param2": mcmc_samples[:, 1]}})

# Calculate Effective Sample Size (ESS)
ess = az.effective_sample_size(idata)
print("Effective Sample Size for each parameter:")
print(ess)