# Chapter 4 — CUDA Device Properties (Eval Task)

This task evaluates your ability to use the CUDA Runtime API to **query GPU device properties** and return them in a structured form. It mirrors the Chapter 4 exercise ("device properties") while being robustly testable in any environment. (Spec: PMPP Ch.4 device query.)

## What you implement

Edit **`student_kernel.cu`** and implement:
```c
int collect_device_info(DeviceInfo* out, int max_out, int* out_count);
```

* Use `cudaGetDeviceCount` and `cudaGetDeviceProperties`.
* Fill every field in `DeviceInfo` (see `student_kernel.cuh`).
* Set `*out_count` to the number of CUDA devices.
* Return `0` on success; non-zero on error.


## How we test

The harness calls your function and the **reference solution** on the same machine, compares **all fields** device-by-device, and passes iff everything matches.

No reliance on stdout formatting—pure data comparison. This guarantees portability across different GPUs.

## Build & Run

```bash
make
make run
```

### Expected output (example)

```
[TEST] PASS: 2 device(s) match reference properties.
```

If it fails, you'll see the first mismatch and an error code.

## Notes

* Works even if there are **0 devices** (both should report 0).
* Based on the Chapter 4 "device properties query" exercise description.