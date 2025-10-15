// test_stencil_shared.cu
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#ifndef IN_TILE_DIM
#define IN_TILE_DIM 8
#endif
#define OUT_TILE_DIM (IN_TILE_DIM-2)

extern __global__ void stencil3d_shared_student(const float*, float*, int, float,float,float,float,float,float,float);

static void cpu_oracle(
    const std::vector<float>& in,
    std::vector<float>& out,
    int N,
    float c0,float c1,float c2,float c3,float c4,float c5,float c6)
{
    auto idx = [N](int I,int J,int K){ return (I*N + J)*N + K; };
    if (N <= 0) return;
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                bool interior = (i>0&&i<N-1)&&(j>0&&j<N-1)&&(k>0&&k<N-1);
                if(!interior){
                    out[idx(i,j,k)] = in[idx(i,j,k)];
                }else{
                    float ctr=in[idx(i,j,k)];
                    float xm=in[idx(i,j,k-1)], xp=in[idx(i,j,k+1)];
                    float ym=in[idx(i,j-1,k)], yp=in[idx(i,j+1,k)];
                    float zm=in[idx(i-1,j,k)], zp=in[idx(i+1,j,k)];
                    out[idx(i,j,k)] = c0*ctr + c1*xm + c2*xp + c3*ym + c4*yp + c5*zm + c6*zp;
                }
            }
        }
    }
}

static void run_case(int N)
{
    const float c0=0.5f, c1=0.1f, c2=0.1f, c3=0.05f, c4=0.05f, c5=0.1f, c6=0.1f;

    size_t count = (N<=0)?0: (size_t)N*N*N;
    std::vector<float> h_in(count), h_cpu(count, 0.0f), h_gpu(count, 1337.0f);

    for (size_t t=0;t<count;t++)
        h_in[t] = float(((t+12345u)*2654435761u) ^ 0xDEADBEEFu) * 1.0e-9f;

    float *d_in=nullptr,*d_out=nullptr;
    cudaMalloc(&d_in, std::max<size_t>(count,1)*sizeof(float));
    cudaMalloc(&d_out,std::max<size_t>(count,1)*sizeof(float));
    cudaMemcpy(d_in, h_in.data(), count*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0xCD, count*sizeof(float));

    // Grid/block for tiled stencil:
    dim3 block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid( (N+OUT_TILE_DIM-1)/OUT_TILE_DIM,
               (N+OUT_TILE_DIM-1)/OUT_TILE_DIM,
               (N+OUT_TILE_DIM-1)/OUT_TILE_DIM );

    // CPU oracle
    cpu_oracle(h_in, h_cpu, N, c0,c1,c2,c3,c4,c5,c6);

    // GPU implementation (student or reference)
    stencil3d_shared_student<<<grid,block>>>(d_in, d_out, N, c0,c1,c2,c3,c4,c5,c6);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu.data(), d_out, count*sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i=0;i<count;i++){
        float rel_err = std::fabs(h_gpu[i]-h_cpu[i]) / (std::fabs(h_cpu[i]) + 1e-6f);
        if (rel_err > 1e-5f){
            fprintf(stderr,"MISMATCH at %zu: got %f, exp %f\n", i, h_gpu[i], h_cpu[i]);
            exit(1);
        }
    }

    // Input immutability
    std::vector<float> h_in_after(count,0.0f);
    cudaMemcpy(h_in_after.data(), d_in, count*sizeof(float), cudaMemcpyDeviceToHost);
    if (h_in_after != h_in){ fprintf(stderr,"Input array was modified!\n"); exit(1); }

    cudaFree(d_in); cudaFree(d_out);
    printf("  N=%d ... OK\n", N);
}

int main(){
    printf("stencil3d-sharedmem-single-turn tests\n");
    int Ns[] = {0,1,2,3,4,6,8,10,16,18,32};
    for (int N: Ns) run_case(N);
    printf("All tests passed.\n");
    return 0;
}