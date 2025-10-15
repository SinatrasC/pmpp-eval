#include "student_kernel.cuh"
#include <cuda_runtime.h>
#include <string.h>

int collect_device_info(DeviceInfo* out, int max_out, int* out_count) {
    // TODO: Implement using CUDA Runtime API
    // Required calls:
    //   - cudaGetDeviceCount(&count)
    //   - For each device id in [0, count): cudaGetDeviceProperties(&prop, id)
    //
    // Required fields to fill in for each DeviceInfo (from cudaDeviceProp prop):
    //   name -> prop.name (ensure null-terminated)
    //   major -> prop.major
    //   minor -> prop.minor
    //   totalGlobalMem -> prop.totalGlobalMem
    //   multiProcessorCount -> prop.multiProcessorCount
    //   totalConstMem -> prop.totalConstMem
    //   sharedMemPerBlock -> prop.sharedMemPerBlock
    //   regsPerBlock -> prop.regsPerBlock
    //   warpSize -> prop.warpSize
    //   maxThreadsPerBlock -> prop.maxThreadsPerBlock
    //   maxThreadsDim{0,1,2} -> prop.maxThreadsDim[0..2]
    //   maxGridSize{0,1,2} -> prop.maxGridSize[0..2]
    //   clockRate -> prop.clockRate
    //   memoryClockRate -> prop.memoryClockRate
    //   memoryBusWidth -> prop.memoryBusWidth
    //   l2CacheSize -> prop.l2CacheSize
    //
    // Return 0 on success, non-zero on failure.

    (void)out; (void)max_out; (void)out_count; // remove after implementing
    return 1; // placeholder: non-zero means "not implemented"
}