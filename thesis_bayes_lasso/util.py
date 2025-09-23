# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:51:56 2024

@author: ichel
"""

"Utility functions to help with plotting and other stuff"

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import jax.numpy as jnp
import numpy as np
import colorsys
from jax import random
import pywt
import jaxwt as jwt

"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

"Remove duplicte labels when plotting multiple vectors"
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')
    
    
    
"Generate nxn matrix with prederfined conditioning"
def create_matrix_cond(n, cond_number, key):
    if n == 1:
        return random.normal(key,[1,1])/np.sqrt(1)/4
    #need orthogonal matrices u,v
    U, _ = np.linalg.qr(random.normal(key,[n, n]))
    V, _ = np.linalg.qr(random.normal(key,[n, n]))

    singular_values = np.linspace(cond_number, 1, n)

    S = np.diag(singular_values)

    A = U @ S @ V.T

    return A

# define filter
def GaussianFilter(s,n): 
    x = np.hstack((np.arange(0,n//2), np.arange(-n//2,0)))
    h = np.exp( (-x**2)/(2*s**2) )
    h = h/sum(h)
    return h

def GaussianFilter_2D(s,n,m): 
    x = np.hstack((np.arange(0,n//2), np.arange(-n//2,0)))
    y = np.hstack((np.arange(0,m//2), np.arange(-m//2,0)))
    [X,Y] = np.meshgrid(y,x)
    h = np.exp( (-X**2-Y**2)/(2*s**2) )
    h = h/sum(sum(h))
    return h

def GaussianFilter_2D_jax(s,n,m): 
    x = jnp.hstack((jnp.arange(0,n//2), jnp.arange(-n//2,0)))
    y = jnp.hstack((jnp.arange(0,m//2), jnp.arange(-m//2,0)))
    [X,Y] = jnp.meshgrid(y,x)
    h = jnp.exp( (-X**2-Y**2)/(2*s**2) )
    h = h/sum(sum(h))
    return h

def flatten_coeffs(coeffs):
    flat_coeffs = []
    coeff_slices = []
    coeff_shapes = []

    current_position = 0

    # Iterate through all levels of the coefficient list
    for coeff in coeffs:
        if isinstance(coeff, dict):  # Handling multi-dimensional coefficients (e.g., 2D wavelets) -- DOESN'T WORK
            for key in coeff:
                flattened = jnp.ravel(coeff[key])
                flat_coeffs.append(flattened)
                coeff_shapes.append(coeff[key].shape)
                coeff_slices.append((current_position, current_position + flattened.size))
                current_position += flattened.size
        else:  # For 1D wavelets
            flattened = jnp.ravel(coeff)
            flat_coeffs.append(flattened)
            coeff_shapes.append(coeff.shape)
            coeff_slices.append((current_position, current_position + flattened.size))
            current_position += flattened.size

    # Concatenate all the flattened coefficients into a single array
    flat_coeffs = jnp.concatenate(flat_coeffs)
    return flat_coeffs, coeff_slices, coeff_shapes

def unflatten_coeffs(flat_coeffs, coeff_slices, coeff_shapes):
    coeffs = []
    for i, (start, end) in enumerate(coeff_slices):
        coeff = flat_coeffs[start:end].reshape(coeff_shapes[i])
        coeffs.append(coeff)
    return coeffs

def flatten_coeffs_2D_OLD(coeffs):
    flat_coeffs = []
    coeff_slices = []
    coeff_shapes = []

    current_position = 0

    # Iterate through all levels of the coefficient list
    # Thought to make it work with dictionaries, similar to original pywt implementation, but jax doesn't like that.
    for coeff in coeffs:
        if isinstance(coeff, dict):  # Handling multi-dimensional coefficients (e.g., 2D wavelets)
            for key in coeff:
                if isinstance(coeff[key], tuple):  # For 2D wavelets
                    for sub_coeff in coeff[key]:
                        flattened = jnp.ravel(sub_coeff)
                        flat_coeffs.append(flattened)
                        coeff_shapes.append(sub_coeff.shape)
                        coeff_slices.append((current_position, current_position + flattened.size))
                        current_position += flattened.size
                else:  # For 1D wavelets - OLD
                    flattened = jnp.ravel(coeff[key])
                    flat_coeffs.append(flattened)
                    coeff_shapes.append(coeff[key].shape)
                    coeff_slices.append((current_position, current_position + flattened.size))
                    current_position += flattened.size
        elif isinstance(coeff, tuple):  # For 2D wavelet coefficients as tuples
            for sub_coeff in coeff:
                flattened = jnp.ravel(sub_coeff)
                flat_coeffs.append(flattened)
                coeff_shapes.append(sub_coeff.shape)
                coeff_slices.append((current_position, current_position + flattened.size))
                current_position += flattened.size
        else:  # For 1D wavelets - OLD
            flattened = jnp.ravel(coeff)
            flat_coeffs.append(flattened)
            coeff_shapes.append(coeff.shape)
            coeff_slices.append((current_position, current_position + flattened.size))
            current_position += flattened.size

    # Concatenate all the flattened coefficients into a single array
    flat_coeffs = jnp.concatenate(flat_coeffs)
    return flat_coeffs, coeff_slices, coeff_shapes

def flatten_coeffs_2D(coeffs):
    flat_coeffs = []
    coeff_slices = []
    coeff_shapes = []

    current_position = 0

    # Iterate through all levels of the coefficient list
    for coeff in coeffs:
        if isinstance(coeff, tuple):  # For 2D wavelet coefficients as tuples
            #print('CASE 1')
            for sub_coeff in coeff:
                flattened = jnp.ravel(sub_coeff)
                flat_coeffs.append(flattened)
                coeff_shapes.append(sub_coeff.shape)
                coeff_slices.append((current_position, current_position + flattened.size))
                current_position += flattened.size
        else:  # For 1D wavelets - Should be first element of a coeffcient list
            #print('CASE 2')
            flattened = jnp.ravel(coeff)
            flat_coeffs.append(flattened)
            coeff_shapes.append(coeff.shape)
            coeff_slices.append((current_position, current_position + flattened.size))
            current_position += flattened.size

    # Concatenate all the flattened coefficients into a single array
    flat_coeffs = jnp.concatenate(flat_coeffs)
    return flat_coeffs, coeff_slices, coeff_shapes

def unflatten_coeffs_2D(flat_coeffs, coeff_slices, coeff_shapes):
    coeffs = []
    counter = 0
    curr_level = []
    for coeff_slice, coeff_shape in zip(coeff_slices, coeff_shapes):
        # Extract the relevant section of the flattened coefficients
        flat_section = flat_coeffs[coeff_slice[0]:coeff_slice[1]]

        #print(f"Unflattening slice from {coeff_slice[0]} to {coeff_slice[1]} with target shape {coeff_shape}")
        #print(f"Flat section shape: {flat_section.shape}")

        # Reshape flat section to match the target shape
        reshaped_coeff = jnp.reshape(flat_section, coeff_shape)
        if counter == 0:
            coeffs.append(reshaped_coeff)
        elif counter == 3:
            curr_level.append(reshaped_coeff)
            coeffs.append(tuple(curr_level))
            curr_level = []
            counter = 0
        else:
            curr_level.append(reshaped_coeff)
            
        
        
        counter += 1

    
    return coeffs



def getWaveletTransforms_jax(n,wavelet_type,level):
    mode = "symmetric"

    
    coeffs_tpl = jwt.wavedec(data=np.zeros(n), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = flatten_coeffs(coeffs_tpl)
    coeffs_tpl_rec = unflatten_coeffs(coeffs_1d, coeff_slices, coeff_shapes)
    
    # Dummy decomposition to extract structure information
    #coeffs_tpl = jwt.wavedec(jnp.zeros(n), wavelet=wavelet_type, level=level, mode=mode)
    #_, coeff_shapes = flatten_coeffs(coeffs_tpl)  # Store shapes for unflattening later

    def py_W(x):
        # Perform wavelet decomposition
        coeffs = jwt.wavedec(x, wavelet=wavelet_type, level=level, mode=mode)
        # Flatten the coefficients manually
        coeffs, _, _ = flatten_coeffs(coeffs)
        return coeffs

    def py_Ws(alpha):
        #coeffs = coeffs.reshape(-1,1)
        # Manually unflatten the coefficients
        coeffs = unflatten_coeffs(alpha, coeff_slices, coeff_shapes)
        # Reconstruct the signal using inverse wavelet transform
        rec = jwt.waverec(coeffs, wavelet=wavelet_type)
        return rec.ravel()
    
    return py_W, py_Ws

"JAXified 2D Wavelet transforms"
def getWaveletTransforms_2D_jax(n,m,wavelet_type,level):
    mode = "symmetric"
    
    coeffs_tpl = jwt.wavedec2(data=jnp.zeros((n, m)), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = flatten_coeffs_2D(coeffs_tpl)
    coeffs_tpl_rec = unflatten_coeffs_2D(coeffs_1d, coeff_slices, coeff_shapes)

    def py_W(im):
        alpha = jwt.wavedec2(data=im, wavelet=wavelet_type, mode=mode, level=level)
        alpha, tstslice, tstshp = flatten_coeffs_2D(alpha)
        return alpha

    def py_Ws(alpha):
        coeffs = unflatten_coeffs_2D(alpha, coeff_slices, coeff_shapes)
        #print(coeffs[0])
        im = jwt.waverec2(coeffs, wavelet=wavelet_type)
        return im[0]
    
    return py_W, py_Ws

"JAXified 2D Wavelet transforms, but higher weights placed on finer coefficients"
def getWaveletTransforms_2D_jax_scale(n,m,wavelet_type,level,weight=1):
    mode = "symmetric"
    
    coeffs_tpl = jwt.wavedec2(data=jnp.zeros((n, m)), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = flatten_coeffs_2D(coeffs_tpl)
    coeffs_tpl_rec = unflatten_coeffs_2D(coeffs_1d, coeff_slices, coeff_shapes)

    print(coeff_slices)
    print(coeff_shapes)

    scaling_vec = jnp.zeros_like(coeffs_1d)
    
    for i, slice in enumerate(coeff_slices):
        # slice is a tuple like (start, end)
        start, end = slice

        if i < 4:  # Only apply weighting to the finer scales (first 4 elements, should correspond to ad, dd, da)
            scaling_vec = scaling_vec.at[start:end].add(weight**i)
        else:
            #apply different scaling to coarse coeffs?
            pass

    def py_W(im):
        alpha = jwt.wavedec2(data=im, wavelet=wavelet_type, mode=mode, level=level)
        alpha, _, _ = flatten_coeffs_2D(alpha)
        return alpha

    def py_Ws(alpha):
        coeffs = unflatten_coeffs_2D(alpha, coeff_slices, coeff_shapes)
        #print(coeffs[0])
        im = jwt.waverec2(coeffs, wavelet=wavelet_type)
        return im[0]
    
    return py_W, py_Ws, scaling_vec

def getWaveletTransforms(n,wavelet_type = "db2",level = 5):
    mode = "periodization"

    
    coeffs_tpl = pywt.wavedec(data=np.zeros(n), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs_tpl)
    coeffs_tpl_rec = pywt.unravel_coeffs(coeffs_1d, coeff_slices, coeff_shapes)

    def py_W(x):
        #print('im in pyW')
        #print(x.shape)
        #x = x.squeeze()
        alpha = pywt.wavedec(data=x, wavelet=wavelet_type, mode=mode, level=level)
        alpha, _, _ = pywt.ravel_coeffs(alpha)
        #print(len(alpha))
        return alpha

    def py_Ws(alpha):
        #print('im in pyWs')
        #print(len(alpha))
        coeffs = pywt.unravel_coeffs(alpha, coeff_slices, coeff_shapes,output_format='wavedec')
        rec = pywt.waverec(coeffs, wavelet=wavelet_type, mode=mode)
        
        #print(rec.shape)
        return rec
    
    return py_W, py_Ws

def getWaveletTransforms_2D(n,m,wavelet_type = "db2",level = 5):
    mode = "periodization"

    
    coeffs_tpl = pywt.wavedecn(data=np.zeros((n, m)), wavelet=wavelet_type, mode=mode, level=level)
    coeffs_1d, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs_tpl)
    coeffs_tpl_rec = pywt.unravel_coeffs(coeffs_1d, coeff_slices, coeff_shapes)

    def py_W(im):
        alpha = pywt.wavedecn(data=im, wavelet=wavelet_type, mode=mode, level=level)
        alpha, _, _ = pywt.ravel_coeffs(alpha)
        return alpha

    def py_Ws(alpha):
        coeffs = pywt.unravel_coeffs(alpha, coeff_slices, coeff_shapes)
        im = pywt.waverecn(coeffs, wavelet=wavelet_type, mode=mode)
        return im
    
    return py_W, py_Ws

def haar_matrix(k):
    if k == 1:
        return jnp.array([[1, 1], [1, -1]])
    else:
        h = haar_matrix(k-1)
    h = jnp.concatenate([   jnp.kron(h,jnp.array([[1,1]]))   ,   jnp.kron(jnp.array(jnp.eye(2**(k-1))),jnp.array([[1,-1]]))   ],0)
    #normalize
    h = h / jnp.sqrt(jnp.linalg.norm(h, ord=1, axis=-1, keepdims=True))
    return h

"Generate toeplitz matrix to apply 2d convolution to flattened image"
def conv_matrix(kernel, image_shape):
    k_size = kernel.shape[0]
    m, n = image_shape
    p = (k_size - 1) // 2  # Calculate the padding
    padded_size = (m + 2*p, n + 2*p)
    
    # Number of elements in the padded image and the output image
    num_elements_padded = padded_size[0] * padded_size[1]
    num_elements_output = m * n
    
    # Initialize the convolution matrix C
    C = np.zeros((num_elements_output, num_elements_padded))
    
    # Fill in the convolution matrix
    for i in range(m):
        for j in range(n):
            row = i * n + j  # Row in the convolution matrix
            
            for ki in range(k_size):
                for kj in range(k_size):
                    padded_i = i + ki
                    padded_j = j + kj
                    col = padded_i * padded_size[1] + padded_j  # Column in the convolution matrix
                    
                    # Place the kernel value in the convolution matrix
                    C[row, col] = kernel[ki, kj]
    
    return jnp.array(C)

"Pad and flatten image - for use with convolution matrix."
def prep_img_conv(img,ker):
    kernel_size = ker.shape[0]
    # Calculate padding
    p = (kernel_size - 1) // 2
    padded_image = np.pad(img, pad_width=p, mode='constant', constant_values=0)
    
    # Flatten the padded image
    return padded_image.flatten()
    