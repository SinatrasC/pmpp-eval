# ch21-bezier-dp-free-child-buffers

Implement a cleanup kernel that **frees** per-line device-heap vertex buffers allocated on the device (via `malloc`), and nulls the pointers. Kernel must be **idempotent** (double-free safe).

## What you implement (student_kernel.cu)
- `__global__ void freeVertexMem(BezierLine* bLines, int nLines)`
  - For each `lidx`: if `vertexPos != nullptr` then `free(vertexPos)` and set to `nullptr`.

## Build
```
make
```

This produces:
- `test_reference` (reference free kernel)
- `test_student`   (your free kernel)

## Run
```
./test_reference
./test_student
```

## Test outline
- A device-side allocator kernel first allocates buffers for each line (`malloc`) and writes dummy data.
- `freeVertexMem` is launched.
- Verify pointers are `nullptr` after copy-back.
- Re-run allocator after free to confirm heap reuse (indirect evidence free worked).
- Launch `freeVertexMem` **twice** to test idempotence.

## Notes
- Uses device `malloc/free` => test sets `cudaLimitMallocHeapSize`.
- No Thrust/Python.