import time
import numpy as np
import jax
import jax.numpy as jnp

# Function to run a matrix computation on the CPU (using NumPy)
def run_cpu_computation(matrix_size):
    # Create a random matrix
    x = np.random.randn(matrix_size, matrix_size)
    
    # Run matrix multiplication
    start_time = time.time()
    y = np.dot(x, x.T)
    end_time = time.time()

    return end_time - start_time  # Return the time taken

# Function to run a matrix computation on the GPU (using JAX)
def run_gpu_computation(matrix_size):
    # Create a random matrix
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (matrix_size, matrix_size))
    
    # Move the computation to the GPU
    x_device = jax.device_put(x, jax.devices()[0])  # Assuming the GPU is the first device
    
    # Run matrix multiplication
    start_time = time.time()
    y = jnp.dot(x_device, x_device.T)
    y.block_until_ready()  # Wait for the computation to complete
    end_time = time.time()

    return end_time - start_time  # Return the time taken

# Test with a larger matrix size (e.g., 5000 x 5000)
matrix_size = 5000
num_runs = 5

# Run the computation multiple times to get a better average time
cpu_times = [run_cpu_computation(matrix_size) for _ in range(num_runs)]
gpu_times = [run_gpu_computation(matrix_size) for _ in range(num_runs)]

print(f"Average time taken on CPU: {np.mean(cpu_times):.4f} seconds")
print(f"Average time taken on GPU: {np.mean(gpu_times):.4f} seconds")