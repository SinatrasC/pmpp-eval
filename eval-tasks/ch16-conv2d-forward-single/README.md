# ch16-conv2d-forward-single

**Task:** 2D Convolution Forward Pass on GPU

**Chapter:** 16 (Deep Neural Networks)

## Problem Description

Implement a **Conv2D forward kernel** for deep neural network inference with NCHW tensor layout.

### Key Concepts
- **NCHW Layout**: Batch, Channels, Height, Width ordering
- **Thread-per-output**: Each thread computes one output element `(n, oc, oh, ow)`
- **Convolution Operation**: Sliding window with learnable filters
- **Optional Bias**: Per-channel bias addition

### Contract
- **Input**: `input[N,C,H,W]`, `weight[OC,C,KH,KW]`, optional `bias[OC]`
- **Output**: `output[N,OC,OH,OW]` where `OH=(H-KH)/SH+1`, `OW=(W-KW)/SW+1`
- **Parameters**: No padding, configurable stride `(SH,SW)`, kernel size `(KH,KW)`
- **Algorithm**:
  1. Decode thread ID to output coordinates `(n,oc,oh,ow)`
  2. Compute input window starting position: `ih0=oh*SH`, `iw0=ow*SW`
  3. Accumulate: `sum(input[n,c,ih,iw] * weight[oc,c,kh,kw])` over valid positions
  4. Add bias if provided: `result = accumulation + bias[oc]`

### Performance Considerations
- **Memory Coalescing**: Threads access consecutive output elements
- **Shared Memory**: Opportunity for input/weight tiling (advanced)
- **Register Usage**: Minimize temporary variables
- **Bounds Checking**: Handle kernel boundaries efficiently

### Files
- `student_kernel.cu`: TODO skeleton for students
- `reference_solution.cu`: Complete working implementation
- `test_conv2d_forward.cu`: Test harness with multiple convolution configurations
- `Makefile`: Build configuration


### Testing
```bash
make test_student    # Build student version
make test_reference  # Build reference version
./test_student       # Run student tests
./test_reference     # Run reference tests (should pass)
```

The test harness validates correctness against CPU convolution oracle and checks for memory safety violations using guard canaries.

### Test Cases
- **1×1×3×3 K3 S1**: Basic 3×3 convolution, no bias
- **1×1×5×5 K3 S2**: Strided convolution, no bias
- **Batch processing**: Multiple samples with bias
- **Multi-channel**: Input/output channel variations
- **Asymmetric**: Different height/width dimensions

Expected output: `Summary: 5/5 passed`
