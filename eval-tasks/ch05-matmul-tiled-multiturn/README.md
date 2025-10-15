# matmul-tiled-multiturn

Shared-memory **tiled matrix multiplication** evaluation task, packaged in the same format as `vecmul-single-turn`.

- Compute: `C[M x K] = A[M x N] * B[N x K]`
- Tile size: **16 x 16**
- Must handle non-multiple dimensions safely (guarded loads/stores)

## Files

- `student_kernel.cu` — **You edit this**. Implement the tiled kernel.
- `reference_solution.cu` — Known-good implementation for validation.
- `test_matmul.cu` — Self-contained test harness with CPU oracle.
- `Makefile` — Build and run targets.

## Build

```bash
make
```

This builds:

* `test_student` (your solution)
* `test_reference` (reference)

## Run

```bash
# Run the reference (should pass all tests)
./test_reference

# Run your student implementation
./test_student
```

## What is required?

Implement the kernel in `student_kernel.cu`:

* TILE size = 16
* Use `__shared__` memory tiles for A and B
* Accumulate across tiles along N
* Guard loads/stores to avoid OOB
* Write C only if `row < M && col < K`

## Test Coverage

The harness:

* Compares results to a CPU oracle (double-accumulation, float output)
* Tests **edge cases** (`0x0x0`, `1x1x1`)
* Tests **non-multiples of 16** (`17x17x17`, `31x37x19`, etc.)
* Uses **adversarial patterns** (zeros, alternating, sequential, sin-like)
* Verifies **input immutability** (A and B must not be modified)

Exit code is **0** on success, **non-zero** on failure.

## Algorithm Overview

**Shared-Memory Tiled Matrix Multiplication:**

1. **Thread Mapping**: Each thread computes one element of output matrix C
2. **Tiling Strategy**: Break large matrices into 16×16 tiles that fit in shared memory
3. **Cooperative Loading**: All threads in a block cooperatively load tiles
4. **Boundary Handling**: Pad out-of-bounds accesses with zeros
5. **Synchronization**: Use `__syncthreads()` to coordinate shared memory access

**Key Implementation Steps:**

```cpp
// 1. Calculate thread's output position
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;

// 2. Declare shared memory tiles
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

// 3. Loop over tiles along inner dimension N
for (int t = 0; t < tiles; ++t) {
    // 4. Cooperatively load A and B tiles (with bounds checking)
    // 5. __syncthreads()
    // 6. Compute partial products using shared data
    // 7. __syncthreads()
}

// 8. Write result to global memory (with bounds checking)
```

This multi-turn task teaches:
- Shared memory usage and bank conflicts
- Thread block cooperation and synchronization  
- Memory coalescing patterns
- Boundary condition handling
- Performance optimization through tiling
