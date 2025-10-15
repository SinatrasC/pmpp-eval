# ch16-maxpool2d-forward-single

**Task:** 2D Max Pooling Forward Pass on GPU

**Chapter:** 16 (Deep Neural Networks)

## Problem Description

Implement a **MaxPool2D forward kernel** for deep neural network inference with NCHW tensor layout.

### Key Concepts
- **NCHW Layout**: Batch, Channels, Height, Width ordering
- **Thread-per-output**: Each thread computes one output element `(n, c, oh, ow)`
- **Max Pooling**: Downsampling operation using maximum value in each window
- **Index Tracking**: Store argmax indices for backward pass support

### Contract
- **Input**: `input[N,C,H,W]` tensor in NCHW format
- **Output**: `output[N,C,OH,OW]` and `indices[N,C,OH,OW]`
- **Dimensions**: `OH=(H-KH)/SH+1`, `OW=(W-KW)/SW+1`
- **Parameters**: No padding, configurable stride `(SH,SW)`, kernel size `(KH,KW)`
- **Algorithm**:
  1. Decode thread ID to output coordinates `(n,c,oh,ow)`
  2. Compute input window: `ih0=oh*SH`, `iw0=ow*SW`
  3. Find maximum: `max(input[n,c,ih,iw])` over `(kh,kw)` in kernel
  4. Store value and local index: `output[idx] = max_val`, `indices[idx] = argmax`

### Index Convention
- **Local indices**: Argmax stored as linear index within KH×KW window
- **Range**: `0` to `KH*KW-1` for position `(kh,kw) = kh*KW + kw`
- **Usage**: Essential for efficient backward pass implementation

### Performance Considerations
- **Memory Coalescing**: Threads access consecutive output elements
- **Register Efficiency**: Minimize temporary storage
- **Bounds Checking**: Handle kernel boundaries correctly
- **Floating-Point**: Use `-FLT_MAX` for initialization

### Files
- `student_kernel.cu`: TODO skeleton for students
- `reference_solution.cu`: Complete working implementation
- `test_maxpool_forward.cu`: Test harness with multiple pooling configurations
- `Makefile`: Build configuration


### Testing
```bash
make test_student    # Build student version
make test_reference  # Build reference version
./test_student       # Run student tests
./test_reference     # Run reference tests (should pass)
```

The test harness validates correctness against CPU max pooling oracle and checks for memory safety violations using guard canaries.

### Test Cases
- **1×1×4×4 K2 S2**: Basic 2×2 pooling with stride 2
- **Batch processing**: Multiple samples and channels
- **3×3 S1**: Overlapping windows
- **Asymmetric**: Different height/width and mixed strides

Expected output: `Summary: 4/4 passed`
