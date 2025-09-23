# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:13:50 2024

@author: ichel
"""

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
import timeit
from PIL import Image
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.numpy.fft import fft2, ifft2
import numpy as np
import torch
import cv2

from skimage.util import random_noise

from runner import Runner
from setup import Setup

#from setup import A, n, M, lam, y, y2, initx, initv
#from algo import Solver
from util import GaussianFilter_2D_jax, getWaveletTransforms_2D_jax, legend_without_duplicate_labels
import arviz as az
import seaborn as sns


'Setup up forward and inverse operators'
#p=1024
#p=32
s = 4
M = 1
#cam = pywt.data.camera()/255
cam = np.array(Image.open(os.path.join(parent_dir, 'lena128.jpg')).convert('L'), dtype=jnp.float64)
cam_norm = cam/255
plt.imshow(cam, cmap="gray")
n,m = cam.shape

blotch_probability = 0.2  # Adjust this value for more or fewer blotches
mask = np.random.choice([0, 1], size=cam.shape, p=[blotch_probability, 1 - blotch_probability])

# Apply the mask to the image (blotches will be set to 0 - black)
blotched_image = cam * mask

# Display the blotched image
plt.subplot(1, 2, 2)
plt.title("Image with Random Blotches")
plt.imshow(blotched_image, cmap='gray')
plt.axis('off')

plt.show()

image_tensor = torch.tensor(cam, dtype=torch.float32) / 255.0

# Create a random mask for blotching (5% of pixels will be masked)
blotch_probability = 0.2  # Adjust this for more/fewer blotches
mask = torch.bernoulli(torch.full(image_tensor.shape, 1 - blotch_probability))

# Apply the mask to the image (masked/blotched areas will be set to 0)
blotched_image_tensor = image_tensor * mask

# Display the blotched image
plt.subplot(1, 2, 2)
plt.title("Grayscale Image with Random Blotches")
plt.imshow(blotched_image_tensor.numpy(), cmap='gray')
plt.axis('off')
plt.show()