#ifndef CH04_DEVICE_PROPS_STUDENT_KERNEL_CUH
#define CH04_DEVICE_PROPS_STUDENT_KERNEL_CUH

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char   name[256];
    int    major;
    int    minor;
    size_t totalGlobalMem;
    int    multiProcessorCount;
    size_t totalConstMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    int    maxThreadsPerBlock;
    int    maxThreadsDim0, maxThreadsDim1, maxThreadsDim2;
    int    maxGridSize0,  maxGridSize1,  maxGridSize2;
    int    clockRate;         // kHz
    int    memoryClockRate;   // kHz
    int    memoryBusWidth;    // bits
    int    l2CacheSize;       // bytes
} DeviceInfo;

/**
 * Fill up to max_out entries of 'out' with device properties.
 * On success, writes the number of devices found to *out_count and returns 0.
 * On CUDA/API failure, returns non-zero. Caller owns the output buffer.
 */
int collect_device_info(DeviceInfo* out, int max_out, int* out_count);

#ifdef __cplusplus
}
#endif

#endif // CH04_DEVICE_PROPS_STUDENT_KERNEL_CUH