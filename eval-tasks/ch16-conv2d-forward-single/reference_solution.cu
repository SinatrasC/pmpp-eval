// ch16-conv2d-forward-single / reference_solution.cu
#include <cuda_runtime.h>

// This is the forward kernel adapted directly to the interface seen in
// code/autograd_manual/src/cudnn/conv2d.cu. Same indexing & contract.
extern "C" __global__
void conv2d_forward_kernel(
    const float* __restrict__ input,      // [N,C,H,W]
    const float* __restrict__ weight,     // [OC,C,KH,KW]
    const float* __restrict__ bias,       // [OC] or nullptr
    float* __restrict__ output,           // [N,OC,OH,OW]
    int N, int C, int H, int W,
    int OC, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int out_h, int out_w)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = 1LL * N * OC * out_h * out_w;
    if ((long long)tid >= total) return;

    int t = tid;
    const int ow = t % out_w;  t /= out_w;
    const int oh = t % out_h;  t /= out_h;
    const int oc = t % OC;     t /= OC;
    const int n  = t;

    const int ih0 = oh * stride_h;
    const int iw0 = ow * stride_w;

    float acc = (bias ? bias[oc] : 0.0f);

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int ih = ih0 + kh; if (ih >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int iw = iw0 + kw; if (iw >= W) continue;

                const long long in_idx =
                    (((long long)n * C + c) * H + ih) * W + iw;
                const long long w_idx =
                    ((((long long)oc * C + c) * kernel_h + kh) * kernel_w + kw);

                acc += input[in_idx] * weight[w_idx];
            }
        }
    }

    const long long out_idx =
        (((long long)n * OC + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = acc;
}