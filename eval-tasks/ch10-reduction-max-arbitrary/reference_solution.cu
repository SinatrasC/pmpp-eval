// reference_solution.cu
#include <cuda_runtime.h>
#include <limits>

__device__ inline
void atomicMaxFloat(float* addr, float val) {
    // Treat NaN as -INF: ignore if val is NaN
    if (isnan(val)) return;
    int*   addr_i = reinterpret_cast<int*>(addr);
    int    old    = *addr_i;
    while (true) {
        float old_f = __int_as_float(old);
        if (old_f >= val) break; // nothing to do
        int assumed = old;
        int desired = __float_as_int(val);
        old = atomicCAS(addr_i, assumed, desired);
        if (old == assumed) break; // success
        // else, someone changed *addr; retry with new old
    }
}

extern "C" __global__
void reduce_max_arbitrary(const float* __restrict__ in,
                          float* __restrict__ out, int n)
{
    extern __shared__ float s[];
    const int tid = threadIdx.x;

    // local max across grid-stride
    long long idx = (long long)blockIdx.x * blockDim.x * 2 + tid;
    const long long stride = (long long)gridDim.x * blockDim.x * 2;

    float local = -INFINITY;
    for (; idx < n; idx += stride) {
        float a = in[idx];
        if (a > local) local = a;
        long long idx2 = idx + blockDim.x;
        if (idx2 < n) {
            float b = in[(int)idx2];
            if (b > local) local = b;
        }
    }

    s[tid] = local;
    __syncthreads();

    // shared-memory max reduction
    for (int step = blockDim.x >> 1; step >= 1; step >>= 1) {
        if (tid < step) {
            float rhs = s[tid + step];
            if (rhs > s[tid]) s[tid] = rhs;
        }
        __syncthreads();
        if (step == 1) break;
    }

    if (tid == 0) atomicMaxFloat(out, s[0]);
}