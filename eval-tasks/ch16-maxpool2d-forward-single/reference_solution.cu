// ch16-maxpool2d-forward-single / reference_solution.cu
#include <cuda_runtime.h>
#include <float.h>

// Signature matches the one found in your conv2d.cu (maxpool2d_forward_kernel).
extern "C" __global__
void maxpool2d_forward_kernel(const float* input, float* output, int* indices,
                              int batch_size, int channels,
                              int height, int width,
                              int kernel_h, int kernel_w,
                              int stride_h, int stride_w,
                              int out_h, int out_w)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = 1LL*batch_size*channels*out_h*out_w;
    if((long long)tid >= total) return;

    int t = tid;
    const int ow = t % out_w;    t /= out_w;
    const int oh = t % out_h;    t /= out_h;
    const int c  = t % channels; t /= channels;
    const int n  = t;

    const int ih0 = oh * stride_h;
    const int iw0 = ow * stride_w;

    float best = -FLT_MAX;
    int best_idx = -1;

    for(int kh=0; kh<kernel_h; ++kh){
        int ih = ih0 + kh; if(ih>=height) continue;
        for(int kw=0; kw<kernel_w; ++kw){
            int iw = iw0 + kw; if(iw>=width) continue;
            const long long in_idx = (((long long)n*channels + c)*height + ih)*width + iw;
            float v = input[in_idx];
            int lid = kh*kernel_w + kw;
            if(v > best){ best = v; best_idx = lid; }
        }
    }

    const long long out_idx = (((long long)n*channels + c)*out_h + oh)*out_w + ow;
    output[out_idx]  = best;
    indices[out_idx] = best_idx;
}