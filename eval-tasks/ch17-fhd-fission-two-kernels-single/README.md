# ch17-fhd-fission-two-kernels-single

**Task:** Implement the loop-fission variant of FᴴD:
1) `compute_mu_kernel` (thread-per-`m`) computes `rMu`, `iMu`.
2) `fhd_accumulate_mu_kernel` (thread-per-`n`) accumulates using precomputed `rMu/iMu`.

## Files
- `student_kernel.cu` – implement both kernels (TODOs inline)
- `reference_solution.cu` – working reference
- `test_fhd_fission.cu` – CPU fused oracle, compares with your fission result
- `Makefile`

## Build & Run
```bash
make
./test_reference
./test_student
```

## Algorithm Description

This task demonstrates loop fission optimization by splitting the FHD computation into two phases:

**Phase 1: Complex Multiplication (`compute_mu_kernel`)**
- Each thread processes one `m` index
- Computes: `rμ[m] = rΦ[m]*rD[m] + iΦ[m]*iD[m]`
- Computes: `iμ[m] = rΦ[m]*iD[m] - iΦ[m]*rD[m]`

**Phase 2: Accumulation (`fhd_accumulate_mu_kernel`)**
- Each thread processes one `n` index
- Uses precomputed `rμ[m]`, `iμ[m]` values
- Performs phase computation and accumulation as in the fused version

## Implementation Notes

* Two separate kernel launches with intermediate storage
* Inputs must be immutable
* Outputs are **added** into (initialized to 0 by tests)
* Accuracy tolerance: `1e-4`
* Results should match the fused implementation exactly