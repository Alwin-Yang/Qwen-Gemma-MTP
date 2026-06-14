# llama.cpp CUDA Build Troubleshooting on RTX 5090

**Environment:** Ubuntu, NVIDIA RTX 5090 (Blackwell, `sm_120a`), Driver 580.159.03

---

## Problem Summary

Building llama.cpp with `-DGGML_CUDA=ON` fails with:

```
nvcc fatal   : Unsupported gpu architecture 'compute_120a'
```

CMake auto-detects the GPU as `sm_120a` (Blackwell/RTX 5090), but the active `nvcc` is too old to support this architecture.

---

## Root Cause Diagnosis

### Step 1 — Check GPU and driver

```bash
nvidia-smi
```

Revealed: **RTX 5090** (Blackwell, `sm_120a`), Driver `580.159.03`, driver-side CUDA support up to **13.0**.

### Step 2 — Check active nvcc

```bash
nvcc --version
```

Revealed: **CUDA 12.0.140** — this is the Ubuntu system package (`nvidia-cuda-toolkit`), installed via `apt`.  
CUDA 12.0 maximum supported architecture: `sm_90` (Hopper). `sm_120a` requires **CUDA ≥ 12.8**.

### Step 3 — Check installed CUDA versions

```bash
ls /usr/local/ | grep cuda
```

Output:
```
cuda
cuda-12
cuda-12.9
cuda-13
cuda-13.0
```

Multiple CUDA versions were already installed, but the symlink `/usr/local/cuda` pointed to `cuda-12`.

### Step 4 — Find why nvcc still resolved to 12.0

```bash
which nvcc     # → /usr/bin/nvcc
echo $PATH     # → no /usr/local/cuda-*/bin prefix
```

`/usr/bin/nvcc` is the system apt package (`nvidia-cuda-toolkit 12.0.140`), which takes priority over `/usr/local/cuda-*/bin` because `/usr/bin` appears in PATH before any cuda path.

### Step 5 — Check if cuda-13.0 nvcc binary exists

```bash
ls /usr/local/cuda-13.0/bin/nvcc
```

Result: **not found** — only config packages were installed for CUDA 13.0, not the full compiler.

---

## Fix

### Step 1 — Install the CUDA 13.0 compiler

```bash
sudo apt install cuda-compiler-13-0
```

Verify:

```bash
/usr/local/cuda-13.0/bin/nvcc --version
# Should show: release 13.0
```

### Step 2 — Update PATH so cuda-13.0 takes priority

Add to `~/.bashrc` (or `~/.zshrc`):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0
```

Apply immediately:

```bash
source ~/.bashrc
which nvcc         # → /usr/local/cuda-13.0/bin/nvcc
nvcc --version     # → release 13.0
```

### Step 3 — Rebuild llama.cpp

```bash
rm -rf llama.cpp/build

cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON

cmake --build llama.cpp/build --config Release -j \
    --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

cp llama.cpp/build/bin/llama-* llama.cpp/
```

CMake will now auto-detect `sm_120a` and compile with CUDA 13.0 — no manual architecture flag needed.

---

## CUDA Architecture Reference

| GPU Series | Architecture | `sm_xx` | Min CUDA Toolkit |
|---|---|---|---|
| RTX 40xx (Ada Lovelace) | Ada | `sm_89` | 11.8 |
| H100 (Hopper) | Hopper | `sm_90` | 12.0 |
| RTX 50xx / B200 (Blackwell) | Blackwell | `sm_120` / `sm_120a` | **12.8** |

---

## Key Lesson

| Component | Command | Notes |
|---|---|---|
| Driver CUDA support | `nvidia-smi` | Upper bound — what the driver can run |
| Installed Toolkit | `nvcc --version` | What actually compiles |
| Active nvcc binary | `which nvcc` | May not be the newest installed version |

`nvidia-smi` shows the **maximum** CUDA version the driver supports, not the version of the installed toolkit. Always cross-check `which nvcc` and `nvcc --version` to confirm the active compiler.

---

## Alternative: Specify Compiler Directly (No PATH Change)

If modifying PATH is undesirable, pass the compiler path explicitly to CMake:

```bash
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
```

