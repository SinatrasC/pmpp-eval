#include "student_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Ref function (implemented in reference_solution.cu)
extern "C" int collect_device_info_ref(DeviceInfo* out, int max_out, int* out_count);

static bool eq_strings(const char* a, const char* b) {
    return std::strncmp(a, b, 255) == 0; // names can vary in length; both are null-terminated
}

static bool eq_device(const DeviceInfo& a, const DeviceInfo& b) {
    return eq_strings(a.name, b.name) &&
           a.major               == b.major &&
           a.minor               == b.minor &&
           a.totalGlobalMem      == b.totalGlobalMem &&
           a.multiProcessorCount == b.multiProcessorCount &&
           a.totalConstMem       == b.totalConstMem &&
           a.sharedMemPerBlock   == b.sharedMemPerBlock &&
           a.regsPerBlock        == b.regsPerBlock &&
           a.warpSize            == b.warpSize &&
           a.maxThreadsPerBlock  == b.maxThreadsPerBlock &&
           a.maxThreadsDim0      == b.maxThreadsDim0 &&
           a.maxThreadsDim1      == b.maxThreadsDim1 &&
           a.maxThreadsDim2      == b.maxThreadsDim2 &&
           a.maxGridSize0        == b.maxGridSize0 &&
           a.maxGridSize1        == b.maxGridSize1 &&
           a.maxGridSize2        == b.maxGridSize2 &&
           a.clockRate           == b.clockRate &&
           a.memoryClockRate     == b.memoryClockRate &&
           a.memoryBusWidth      == b.memoryBusWidth &&
           a.l2CacheSize         == b.l2CacheSize;
}

int main() {
    const int MAX_DEV = 32;
    DeviceInfo refBuf[MAX_DEV], stuBuf[MAX_DEV];
    int refCount = -1, stuCount = -2;

    int rc_ref = collect_device_info_ref(refBuf, MAX_DEV, &refCount);
    if (rc_ref != 0) {
        std::fprintf(stderr, "[TEST] Reference failed with code %d\n", rc_ref);
        return 100 + rc_ref;
    }

    int rc_stu = collect_device_info(stuBuf, MAX_DEV, &stuCount);
    if (rc_stu != 0) {
        std::fprintf(stderr, "[TEST] Student function returned error %d\n", rc_stu);
        return 200 + rc_stu;
    }

    if (stuCount != refCount) {
        std::fprintf(stderr, "[TEST] Device count mismatch: student=%d, reference=%d\n", stuCount, refCount);
        return 3;
    }

    // If no devices, we still pass when counts match (both 0)
    for (int i = 0; i < stuCount; ++i) {
        if (!eq_device(stuBuf[i], refBuf[i])) {
            std::fprintf(stderr, "[TEST] Mismatch at device %d\n", i);
            std::fprintf(stderr, "  Student name:   %s\n", stuBuf[i].name);
            std::fprintf(stderr, "  Reference name: %s\n", refBuf[i].name);
            // You can print more fields here if you like for debugging
            return 4;
        }
    }

    std::printf("[TEST] PASS: %d device(s) match reference properties.\n", stuCount);
    return 0;
}