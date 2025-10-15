// ch16-maxpool2d-forward-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void maxpool2d_forward_kernel(const float* input, float* output, int* indices,
                              int batch_size, int channels,
                              int height, int width,
                              int kernel_h, int kernel_w,
                              int stride_h, int stride_w,
                              int out_h, int out_w)
{
    (void)input;
    (void)batch_size; (void)channels;
    (void)height; (void)width;
    (void)kernel_h; (void)kernel_w;
    (void)stride_h; (void)stride_w;
    (void)out_h; (void)out_w;

    // TODO: For each thread, compute the maximum value within the pooling
    // window and track the argmax (flattened kernel index).

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 1LL * batch_size * channels * out_h * out_w;
    if ((long long)tid >= total) {
        return;
    }

    if (output) {
        output[tid] = 0.0f;
    }
    if (indices) {
        indices[tid] = -1;
    }
}
