// ch16-conv2d-forward-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OC, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int out_h, int out_w)
{
    (void)input;
    (void)weight;
    (void)bias;
    (void)N; (void)C; (void)H; (void)W;
    (void)OC; (void)kernel_h; (void)kernel_w;
    (void)stride_h; (void)stride_w;
    (void)out_h; (void)out_w;

    // TODO: Map each thread to a unique (n, oc, oh, ow) output element and
    // perform the convolution accumulation over the input tensor.

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 1LL * N * OC * out_h * out_w;
    if ((long long)tid >= total) {
        return;
    }

    if (output) {
        output[tid] = 0.0f; // placeholder value so the stub compiles
    }
}
