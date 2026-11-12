# PyKeOps CUDA & GPU Setup Guidelines (Ubuntu)

This guide provides step-by-step instructions for resolving missing CUDA libraries and configuring **PyKeOps** for GPU acceleration on Ubuntu Linux systems.

---

## 1. Problem Overview

PyTorch packages its own CUDA binaries, allowing `torch.cuda.is_available()` to return `True`. However, **PyKeOps compiles C++/CUDA code at runtime** using system tools (CMake, C++ compilers, `nvcc`, NVRTC). 

If system CUDA headers (`cuda.h`) or compilation libraries (`libnvrtc.so`) are not found by the system linker (`ld`), KeOps falls back to CPU mode with the following warning:

```text
[KeOps] Warning : CUDA libraries not found or could not be loaded; Switching to CPU only.
/usr/bin/x86_64-linux-gnu-ld.bfd: cannot find -lnvrtc: No such file or directory
```

---

## 2. Global System Setup (No Python Script Modifications)

To fix the `-lnvrtc` linker issue globally across all Python virtual environments without adding environment variables to your codebase:

### Step 1: Create a System Symbolic Link for `libnvrtc.so`
Link the CUDA NVRTC shared library to standard system library directories (`/usr/lib`):

```bash
sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so /usr/lib/libnvrtc.so
sudo ldconfig
```

### Step 2: Ensure Required Build Dependencies Are Installed
Verify that `nvcc` and host C++ compilers are installed:

```bash
sudo apt update
sudo apt install -y cuda-toolkit build-essential cmake g++
```

---

## 3. Environment Variable Configuration (Alternative Method)

If system-wide symlinking is not preferred, configure the environment variables in your shell (`~/.bashrc`) or virtual environment (`.venv/bin/activate`):

```bash
# CUDA Base Path
export CUDA_PATH=/usr/local/cuda-13.1

# Executable Path (nvcc)
export PATH=/usr/local/cuda-13.1/bin:$PATH

# Linker Search Paths (libnvrtc.so)
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CXX=g++
```

---

## 4. Verifying KeOps GPU Visibility in Python

Run this script to verify that PyKeOps correctly compiles CUDA kernels and detects your GPU:

```python
import pykeops
import torch

# 1. Check PyTorch CUDA Detection
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# 2. Check PyKeOps GPU Flag
print(f"KeOps GPU available: {pykeops.config.gpu_available}")

# 3. Test Compilation & PyTorch Bindings
pykeops.test_torch_bindings()
```

### Expected Output
```text
PyTorch CUDA available: True
GPU Device: [Your GPU Model]
KeOps GPU available: True
[KeOps] Compiling cuda jit compiler engine ... OK
[pyKeOps] Compiling nvrtc binder for python ... OK
pyKeOps with torch bindings is working!
```

---

## 5. Troubleshooting & Maintenance

* **Clear KeOps Build Cache:**  
  If compilation errors persist after changing configurations, purge the build cache:
  ```bash
  rm -rf ~/.cache/pykeops*
  ```

* **Verify NVRTC Location:**  
  To verify where `libnvrtc.so` lives on your system:
  ```bash
  find /usr/local/cuda* -name "libnvrtc.so" 2>/dev/null
  ```