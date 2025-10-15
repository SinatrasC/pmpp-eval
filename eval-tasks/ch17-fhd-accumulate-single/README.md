# ch17-fhd-accumulate-single

**Task:** Implement the fused FᴴD accumulation kernel in CUDA (one thread per `n`, loop over `m`).

## Files
- `student_kernel.cu` – implement `fhd_accumulate_kernel` (TODOs inline)
- `reference_solution.cu` – working reference
- `test_fhd_accumulate.cu` – CPU oracle + guards + deterministic tests
- `Makefile`

## Build & Run
```bash
make
./test_reference   # should pass all
./test_student     # passes when your kernel is correct
```

## Algorithm Description

The FHD (Fast Hankel Transform with Density) accumulation computes:

For each output point `n`, accumulate contributions from all `m` points:
1. Compute complex products: `rμ = rΦ*rD + iΦ*iD`, `iμ = rΦ*iD - iΦ*rD`
2. Compute phase: `θ = 2π(kx*x + ky*y + kz*z)`
3. Apply rotation and accumulate: `FHD += μ * e^(iθ)`

## Implementation Notes

* Each thread processes one output point `n`
* Loop over all `M` input points for accumulation
* Preserve input arrays (immutability)
* Output buffers may be pre-initialized; your kernel should **add** to them
* Accuracy tolerance: `1e-4`
* Use `cosf()` and `sinf()` for fast trigonometric functions