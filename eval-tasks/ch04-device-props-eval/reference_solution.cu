#include "student_kernel.cuh"
#include <cuda_runtime.h>
#include <string.h>

static void copy_prop(DeviceInfo& dst, const cudaDeviceProp& src) {
    memset(&dst, 0, sizeof(DeviceInfo));
    // name
    strncpy(dst.name, src.name, sizeof(dst.name) - 1);
    dst.name[sizeof(dst.name)-1] = '\0';
    // numerics
    dst.major               = src.major;
    dst.minor               = src.minor;
    dst.totalGlobalMem      = src.totalGlobalMem;
    dst.multiProcessorCount = src.multiProcessorCount;
    dst.totalConstMem       = src.totalConstMem;
    dst.sharedMemPerBlock   = src.sharedMemPerBlock;
    dst.regsPerBlock        = src.regsPerBlock;
    dst.warpSize            = src.warpSize;
    dst.maxThreadsPerBlock  = src.maxThreadsPerBlock;
    dst.maxThreadsDim0      = src.maxThreadsDim[0];
    dst.maxThreadsDim1      = src.maxThreadsDim[1];
    dst.maxThreadsDim2      = src.maxThreadsDim[2];
    dst.maxGridSize0        = src.maxGridSize[0];
    dst.maxGridSize1        = src.maxGridSize[1];
    dst.maxGridSize2        = src.maxGridSize[2];
    dst.clockRate           = src.clockRate;
    dst.memoryClockRate     = src.memoryClockRate;
    dst.memoryBusWidth      = src.memoryBusWidth;
    dst.l2CacheSize         = src.l2CacheSize;
}

extern "C" int collect_device_info_ref(DeviceInfo* out, int max_out, int* out_count) {
    if (!out || !out_count || max_out <= 0) return 2;

    int count = 0;
    cudaError_t st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess) return 3;

    *out_count = count;
    int to_copy = (count < max_out) ? count : max_out;

    for (int dev = 0; dev < to_copy; ++dev) {
        cudaDeviceProp prop{};
        st = cudaGetDeviceProperties(&prop, dev);
        if (st != cudaSuccess) return 4;
        copy_prop(out[dev], prop);
    }
    return 0;
}