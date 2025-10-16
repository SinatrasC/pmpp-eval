# PMPP Evaluation Tasks (eval-tasks)

## Overview

This directory contains 53 CUDA programming evaluation tasks based on the "Programming Massively Parallel Processors" (PMPP) textbook by Hwu, Kirk, and Hajj. Each task evaluates a specific CUDA programming concept or optimization technique.

**Source:** [SinatrasC/pmpp-eval](https://github.com/SinatrasC/pmpp-eval)

---

## Directory Structure

Each evaluation task follows a standardized structure:

```
eval-tasks/
├── ch02-vecadd-single-turn/
│   ├── Makefile                    # Build configuration
│   ├── README.md                   # Task documentation
│   ├── student_kernel.cu           # Student implementation file (to be completed)
│   ├── reference_solution.cu       # Reference implementation
│   └── test_student.cu             # Test harness
├── ch02-vecmul-single-turn/
│   └── ...
├── ch03-ex1a-matmul-row-per-thread/
│   └── ...
└── ... (53 tasks total)
```

### Standard Files in Each Task

| File | Purpose |
|------|---------|
| `Makefile` | Defines build targets (`test_student`, `test_reference`) |
| `README.md` | Task description, requirements, and hints |
| `student_kernel.cu` | Skeleton file where students implement the CUDA kernel |
| `reference_solution.cu` | Correct reference implementation |
| `test_*.cu` | Test harness that validates correctness |

---

## Task Categories by Chapter

### Chapter 2: Data Parallelism
- `ch02-vecadd-single-turn` - Vector addition
- `ch02-vecmul-single-turn` - Vector multiplication

### Chapter 3: Multidimensional Grids and Data
- `ch03-ex1a-matmul-row-per-thread` - Matrix multiplication (row-per-thread)
- `ch03-ex1b-matmul-col-per-thread` - Matrix multiplication (column-per-thread)
- `ch03-rgb2gray-single-turn` - RGB to grayscale conversion

### Chapter 4: Memory Architecture and Performance
- `ch04-device-props-eval` - Device properties evaluation
- `ch04-matmul-basic-single-turn` - Basic matrix multiplication

### Chapter 5: Shared Memory and Tiling
- `ch05-matmul-tiled` - Tiled matrix multiplication
- `ch05-matmul-tiled-multiturn` - Multi-turn tiled matrix multiplication
- `ch05-matmul-tiled-speed` - Optimized tiled matrix multiplication

### Chapter 6: Performance Optimization
- `ch06-thread-coarsening-matmul` - Thread coarsening in matrix multiplication

### Chapter 7: Convolution
- `ch07-conv1d-basic-single-turn` - 1D convolution (basic)
- `ch07-conv1d-tiled-caching` - 1D convolution with tiled caching
- `ch07-conv2d-basic` - 2D convolution (basic)
- `ch07-conv2d-tiled-constant` - 2D convolution with constant memory

### Chapter 8: Stencil Computations
- `ch08-stencil-1d-basic` - 1D stencil computation
- `ch08-stencil-2d-basic` - 2D stencil computation

### Chapter 9: Parallel Histogram
- `ch09-histogram-naive-single-turn` - Naive histogram (global atomics)
- `ch09-histogram-privatization` - Histogram with privatization

### Chapter 10: Reduction
- `ch10-reduction-max-arbitrary` - Reduction (max) with arbitrary size
- `ch10-reduction-sum-2048` - Reduction (sum) for 2048 elements
- `ch10-reduction-sum-arbitrary` - Reduction (sum) with arbitrary size

### Chapter 11: Prefix Sum (Scan)
- `ch11-prefix-sum-kogge-stone` - Kogge-Stone scan
- `ch11-prefix-sum-brent-kung` - Brent-Kung scan

### Chapter 12: Merge
- `ch12-merge-basic` - Basic merge
- `ch12-merge-tiled` - Tiled merge

### Chapter 13: Sorting
- `ch13-bitonic-sort` - Bitonic sort
- `ch13-radix-sort-basic` - Basic radix sort

### Chapter 14: Sparse Matrix
- `ch14-spmv-coo` - Sparse matrix-vector multiply (COO format)
- `ch14-spmv-csr` - Sparse matrix-vector multiply (CSR format)
- `ch14-spmv-ell` - Sparse matrix-vector multiply (ELL format)

### Chapter 15: Graph Search
- `ch15-bfs-direction-optimized-single` - BFS with direction optimization
- `ch15-bfs-edge-centric-single` - Edge-centric BFS
- `ch15-bfs-pull-single` - Pull-based BFS
- `ch15-bfs-push-single` - Push-based BFS

### Chapter 16: Deep Learning
- `ch16-softmax-basic` - Basic softmax
- `ch16-layernorm-basic` - Basic layer normalization

### Chapter 17: Iterative Methods
- `ch17-sparse-iterative-cg` - Conjugate gradient method

### Chapter 18: Parallel Patterns
- `ch18-segmented-scan` - Segmented scan

### Chapter 19: Advanced Optimization
- `ch19-warp-shuffle-reduction` - Warp shuffle reduction
- `ch19-warp-vote-predicate` - Warp vote predicates

### Chapter 20: CUDA Streams
- `ch20-streams-overlap` - Stream-based overlap

### Chapter 21: Dynamic Parallelism
- `ch21-bezier-dp-free-child-buffers` - Bezier curve with dynamic parallelism
- `ch21-bezier-dp-parent-child-single` - Parent-child dynamic parallelism
- `ch21-quadtree-dp-build-single` - Quadtree with dynamic parallelism
- `ch21-quadtree-dp-pack-coalesced` - Coalesced quadtree packing

---

## Build System

Each task uses a `Makefile` with standard targets:

### Build Targets
```bash
make test_student    # Build and run student implementation
make test_reference  # Build and run reference solution
make clean          # Clean build artifacts
```

### Common Makefile Variables
- `NVCC` - NVIDIA CUDA compiler (default: `nvcc`)
- `NVCC_FLAGS` - Compiler flags (e.g., `-arch=sm_70`, `-O3`)
- `CUDA_PATH` - CUDA installation path

---

## Evaluation Process

### Local Evaluation
```bash
# Navigate to task directory
cd eval-tasks/ch02-vecadd-single-turn/

# Edit student_kernel.cu with your implementation
vim student_kernel.cu

# Build and test
make test_student

# Compare with reference
make test_reference
```

### Automated Evaluation
The PMPP evaluation harness automatically:
1. Extracts CUDA code from LLM responses
2. Writes code to `student_kernel.cu`
3. Compiles using `make test_student`
4. Runs the test binary
5. Reports success/failure (1.0 or 0.0)

---

## Task Format

### Student Implementation
Students must complete the skeleton in `student_kernel.cu`:

```cuda
__global__ void myKernel(float* input, float* output, int n) {
    // TODO: Implement kernel
    // Hints provided in comments
}
```

### Test Harness
Each test harness (`test_*.cu`):
- Allocates input/output buffers
- Initializes test data
- Launches student kernel
- Validates results against expected output
- Returns exit code 0 (success) or 1 (failure)

---

## Requirements

### Software
- CUDA Toolkit 11.0+ (nvcc compiler)
- GNU Make
- C++14 or later
- Linux/WSL2 (recommended)

### Hardware
- NVIDIA GPU with compute capability 5.0+
- Recommended: 4GB+ VRAM

---

## Task Naming Convention

Format: `chXX-topic-variant`

- `chXX` - Chapter number (02-21)
- `topic` - Task topic (e.g., vecadd, matmul, histogram)
- `variant` - Task variant:
  - `single-turn` - Single-turn task
  - `multiturn` - Multi-turn task
  - `basic` - Basic implementation
  - `tiled` - Tiled/optimized version
  - `dp` - Dynamic parallelism

---

## Dataset Integration

Tasks are referenced in `pmpp/datasets/pmpp_coding.jsonl`:

```json
{
  "type": "coding",
  "id": "ch02-vecadd-single-turn",
  "question": "Task: ch02-vecadd-single-turn\n...",
  "task_dir": "eval-tasks/ch02-vecadd-single-turn",
  "student_file": "student_kernel.cu",
  "student_targets": ["test_student"],
  "timeout_sec": 180
}
```

---

## Download and Caching

### Automatic Download
Tasks are automatically downloaded from GitHub releases on first use:

```bash
# Default: Downloads to ~/.cache/pmpp/eval-tasks
uv run vf-eval pmpp -m openai/gpt-4o-mini -n 5

# Custom cache location
uv run vf-eval pmpp -n 5 \
  --env-args '{"eval_tasks_cache_dir": "/custom/path"}'
```

### Manual Download
```bash
# Download specific version
wget https://github.com/SinatrasC/pmpp-eval/releases/download/v1.0.0/eval-tasks.tar.gz

# Extract
tar -xzf eval-tasks.tar.gz

# Use in evaluation
uv run vf-eval pmpp -n 5 \
  --env-args '{"use_bundled_tasks": true}'
```

---

## Task Statistics

- **Total Tasks:** 53 CUDA coding tasks
- **Difficulty Range:** Beginner to advanced
- **Average Completion Time:** 2-5 minutes per task (for LLMs)
- **Compilation Time:** ~1-3 seconds per task
- **Execution Time:** <1 second per test

---

## Common Issues

### Issue: Missing CUDA Toolkit
```bash
# Check CUDA installation
nvcc --version

# Install CUDA (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit
```

### Issue: Compilation Errors
- Ensure correct CUDA architecture flags in Makefile
- Check compute capability: `nvidia-smi`
- Verify C++ standard compatibility

### Issue: Runtime Errors
- Check GPU memory availability
- Validate kernel launch parameters
- Review synchronization points

---

## Contributing

To add new evaluation tasks:

1. Create task directory: `eval-tasks/chXX-topic-variant/`
2. Add required files (Makefile, README, student_kernel.cu, etc.)
3. Write test harness with clear pass/fail criteria
4. Update dataset JSONL with task metadata
5. Test with reference implementation
6. Submit PR to [pmpp-eval repository](https://github.com/SinatrasC/pmpp-eval)

---

## License

Evaluation tasks are distributed under the same license as the PMPP codebase.

---

## Support

- **Issues:** [GitHub Issues](https://github.com/SinatrasC/pmpp-eval/issues)
- **Documentation:** [PMPP Environment README](https://github.com/SinatrasC/prime-environments/tree/pmpp)
- **Author:** Sinatras - [GitHub](https://github.com/SinatrasC) · [X](https://x.com/myainotez)

---

## References

- **Textbook:** "Programming Massively Parallel Processors" by Hwu, Kirk, and Hajj
- **CUDA Programming Guide:** [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **GitHub Repository:** [SinatrasC/pmpp-eval](https://github.com/SinatrasC/pmpp-eval)

