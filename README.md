# Homework 3: JIT Compilation and Tracing for ML on GPUs

## Overview
This repository contains the completed submission for HW3, covering JIT compilation
concepts in JAX and PyTorch including compilation overhead, shape specialization,
operator fusion, backend comparison, and graph capture.

## Repository Contents
| File | Description |
|---|---|
| `Homework3_Questions.ipynb` | Conceptual questions (Section 1) |
| `jax_jit_analysis.ipynb` | JAX JIT deep dive (Section 2) |
| `torch_compile_analysis.ipynb` | PyTorch compile analysis (Section 3) |
| `Homework3_Questions - Colab.pdf` | PDF export of conceptual answers |
| `jax_jit_analysis - Colab.pdf` | PDF export of JAX notebook |
| `torch_compile_analysis - Colab.pdf` | PDF export of PyTorch notebook |

## Requirements
```bash
pip install jax jaxlib torch torchvision matplotlib
```

## Running the Code
All notebooks are designed to run in **Google Colab** with a GPU runtime.

1. Open each `.ipynb` file in Google Colab
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order: **Runtime → Run all**

## Notes
- JAX version: 0.7.2
- PyTorch version: 2.10.0+cu128
- All experiments were run on a CUDA GPU (CudaDevice id=0)
