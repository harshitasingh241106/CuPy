# CuPy - GPU Accelerated NumPy for CUDA

## Introduction
CuPy is an open-source library that accelerates numerical computations by utilizing GPU processing with CUDA. It is designed to work seamlessly with NumPy, allowing users to run matrix operations and scientific computations much faster than on a CPU. This blog introduces CuPy's key features, installation, and real-world applications to help you understand how to leverage its power in performance-critical tasks.

---

## Installation & Setup
To download the compatible version of CuPy on your device, and install it; Open the terminal and enter:
```bash
pip install cupy-cuda12x
```
To install CuPy using anaconda :
```bash
conda install -c conda-forge cupy
```
To verify the installation in jupyter notebook, run the following code : 
```bash
import cupy as cp 
print(cp.__version__)
```
If the installation is successful, it will print the CuPy version.
Handling installation issues - If CuPy fails to detect your GPU, ensure you have NVIDIA drivers and CUDA Toolkit installed.
If using Google Colab/ Jupyter notebook, install CuPy with:
```bash
!pip install cupy-cuda12x
```

Ensure that you have an NVIDIA GPU with CUDA installed. You can check your CUDA version with:
```bash
nvcc --version
```

---

## Key Features & Explanation
### 1. NumPy and SciPy Compatibility
CuPy is highly compatible with NumPy and SciPy, meaning you can easily transition to GPU acceleration by simply replacing `numpy` with `cupy` and `scipy` with `cupyx.scipy` in your code.

### 2. GPU Accelerated with CUDA
CuPy leverages CUDA to accelerate computations using the GPU. GPUs excel at parallel processing, allowing thousands of operations to be executed simultaneously. This makes CuPy significantly faster, especially for large datasets or complex matrix operations.

### 3. Efficient Memory Management
CuPy uses memory pools to manage GPU memory efficiently, reducing memory allocation overhead and minimizing CPU-GPU synchronization delays.

- **Device Memory Pool**: Optimizes GPU memory allocation.
- **Pinned Memory Pool**: Manages non-swappable CPU memory, improving data transfer efficiency between CPU and GPU.

### 4. Custom CUDA Kernels
CuPy allows you to create custom CUDA kernels with minimal C++ code to enhance performance. The kernels are compiled and cached for reuse, saving time in future executions.

### 5. Fast Matrix Operations
CuPy performs matrix operations like multiplication, inversion, and eigenvalue decomposition much faster than NumPy. These operations are essential for fields such as scientific computing, machine learning, and numerical simulations.

---

## Code Examples
### 1. Basic Array operations
```python
import cupy as cp

# Creating arrays
x_gpu = cp.array([1, 2, 3, 4, 5])
y_gpu = cp.array([5, 4, 3, 2, 1])

# Element-wise addition
z_gpu = x_gpu + y_gpu

print(z_gpu)
```
![Basic Array](https://github.com/harshitasingh241106/CuPy/blob/main/Screenshot%202025-02-23%20215428.png)

### 2. Comparing NumPy vs CuPy Performance 
CuPy is significantly faster for large-scale operations

```python
import numpy as np
import cupy as cp
import time

size = 10**7  # Large array size

# NumPy (CPU)
x_cpu = np.random.rand(size)
start = time.time()
np_result = np.sqrt(x_cpu)  # Compute square root
end = time.time()
print(f"NumPy Time: {end - start:.5f} seconds")

# CuPy (GPU)
x_gpu = cp.random.rand(size)
start = time.time()
cp_result = cp.sqrt(x_gpu)  # Compute square root
cp.cuda.Device(0).synchronize()  # Ensure GPU computation finishes
end = time.time()
print(f"CuPy Time: {end - start:.5f} seconds")
```
![Comparision](https://github.com/harshitasingh241106/CuPy/blob/main/Screenshot%202025-02-23%20215428.png)

### 3. Matrix Multiplication
GPU-accelerated matrix multiplication is much faster than CPU-based NumPy.
```python
import cupy as cp

# Creating random matrices
A = cp.random.rand(1000, 1000)
B = cp.random.rand(1000, 1000)

# GPU matrix multiplication
C = cp.dot(A, B)

print(C.shape)
```
![Basic Array](https://github.com/harshitasingh241106/CuPy/blob/main/Screenshot%202025-02-23%20215501.png)

### 4. Moving Data Between NumPy and CuPy

Use cp.asnumpy() to move data to CPU and cp.asarray() to move it to GPU.
```python
import cupy as cp
import numpy as np

# Create CuPy array
x_gpu = cp.array([1, 2, 3, 4, 5])

# Convert to NumPy (CPU)
x_cpu = cp.asnumpy(x_gpu)

# Convert back to CuPy (GPU)
x_gpu_again = cp.asarray(x_cpu)

print(type(x_cpu))
print(type(x_gpu_again))

```
![Basic Array](https://github.com/harshitasingh241106/CuPy/blob/main/Screenshot%202025-02-23%20215537.png)

### 5. Using CuPy for Element-Wise Custom Kernels (CUDA Acceleration)

```python
import cupy as cp

# Custom CUDA kernel
ker = cp.ElementwiseKernel(
    'float32 x',      # Input argument(s)
    'float32 y',      # Output argument(s)
    'y = x * x;',     # Operation (square each element)
    'square_kernel'   # Kernel name
)

# Create CuPy array
x_gpu = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)

# Apply custom kernel
y_gpu = ker(x_gpu)

print(y_gpu)


```
![Basic Array](https://github.com/harshitasingh241106/CuPy/blob/main/Screenshot%202025-02-23%20215617.png)

---

## Use Cases & Applications
### 1. Accelerating AI & Deep Learning
CuPy accelerates the training of neural networks by speeding up matrix operations and gradient computations, reducing the time it takes to train models in frameworks like TensorFlow and PyTorch.

**Example:** AI research teams use CuPy to train complex models, such as Convolutional Neural Networks (CNNs), in less time.

### 2. Scientific Simulations
Researchers in fields such as quantum physics and astrophysics can simulate physical systems faster using CuPy, which enables more complex calculations in less time.

**Example:** A molecular biologist uses CuPy to speed up protein interaction simulations, leading to faster discoveries in biological processes.

### 3. Big Data & High-Performance Computing
CuPy speeds up big data processing, making real-time analytics and decision-making more efficient.

**Example:** A financial company uses CuPy to analyze stock market data in real time, providing instant insights.

### 4. Image Processing & Computer Vision
CuPy accelerates image and signal processing tasks such as filtering, edge detection, and Fourier transforms.

**Example:** A startup uses CuPy to process MRI scans faster, improving diagnostic speed.

---

## Conclusion
CuPy is an essential tool for leveraging GPU acceleration in numerical computations. Its compatibility with NumPy makes it easy to integrate, and its CUDA-powered speed improvements make it ideal for AI, scientific simulations, and large-scale data processing.

If you're working with performance-critical applications, CuPy is a great alternative to CPU-based computations.

## Key Takeaways
--> Faster computations with GPU acceleration  
--> Easy integration with NumPy-based projects  
--> Ideal for machine learning, simulations, and data science  

## References & Further Reading
- [CuPy Official Docs](https://docs.cupy.dev/en/stable/)  
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  

