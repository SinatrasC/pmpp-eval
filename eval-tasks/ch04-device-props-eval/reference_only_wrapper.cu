// reference_only_wrapper.cu - For reference-only builds
// This provides the student function by aliasing it to the reference
#include "student_kernel.cuh"

// Forward declare the reference function
extern "C" int collect_device_info_ref(DeviceInfo* out, int max_out, int* out_count);

// Provide student function that calls reference (for reference-only testing)
extern "C" int collect_device_info(DeviceInfo* out, int max_out, int* out_count) {
    return collect_device_info_ref(out, max_out, out_count);
}