# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:10:19 2024

@author: ichel
"""

import jax
import jax.numpy as jnp

# Create a JAX array on the default device (likely GPU if available)
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))

# Move the array to CPU memory
x_cpu = jax.device_put(x, device=jax.devices("cpu")[0])

# Convert to a NumPy array using jax.device_get
x_numpy = jax.device_get(x_cpu)

# Now x_numpy is a NumPy array that resides in CPU memory
print(x_numpy)

print(jax.devices("cpu"))
print(jax.devices("gpu"))