# Chapter 0 — Multi-Architecture Build Example

This toy task shows how to compile CUDA test harnesses for multiple GPU architectures without editing every Makefile.

## Build

```
# Builds both binaries targeting sm_70, sm_80, sm_90 by default
make

# Override the target list at build time
make SM_LIST="70 75 80 90"
```

`SM_LIST` enumerates the compute capabilities you want. The Makefile turns that list into matching `-gencode` flags and also emits PTX for the highest entry so newer drivers can JIT on forward-compatible GPUs.

## Files
- `Makefile` — multi-arch build logic
- `test_demo.cu` — lightweight harness that compares student vs reference addition
- `reference_solution.cu` — correct CUDA kernel
- `student_kernel.cu` — placeholder kernel (intentionally incomplete)

Running `./test_reference` should pass on any GPU with compute capability equal to or above the smallest entry in `SM_LIST`. The student binary fails until the kernel is implemented.
