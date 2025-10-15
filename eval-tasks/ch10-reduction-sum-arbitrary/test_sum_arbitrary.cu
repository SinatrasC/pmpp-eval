// test_sum_arbitrary.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" __global__
void reduce_sum_arbitrary(const float* in, float* out, int n);

static void check(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }
static bool almost_equal(float a,float b,float eps=1e-4f){
    float d=std::fabs(a-b);
    return d<=eps || d<=eps*std::max(1.f,std::max(std::fabs(a),std::fabs(b)));
}
static double cpu_sum(const std::vector<float>& v){ long double s=0; for(float x:v) s+=x; return (double)s; }

static void fill_pattern(std::vector<float>& v, int pat){
    for(size_t i=0;i<v.size();++i){
        switch(pat%4){
            case 0: v[i] = 1.0f; break;
            case 1: v[i] = (float)((13*i)%313) * 0.01f - 1.0f; break;
            case 2: v[i] = (i%2? -2.0f: +2.0f); break;
            default:v[i] = 0.001f*(float)i - 0.25f; break;
        }
    }
}

static int pick_grid(int n, int b){
    if(n<=0) return 1;
    int blocks = (n + (2*b) - 1) / (2*b);
    return std::min(std::max(blocks,1), 120);
}

int main(){
    printf("reduction-sum-arbitrary tests\n");
    const int sizes[] = {0,1,2,31,32,1000,2048,4097,50000};
    const int blocksizes[] = {128,256,512};

    const size_t GUARD=4096; const float SENT=1337.0f;
    int total=0, passed=0;

    for(int n : sizes){
      for(int pat=0; pat<4; ++pat){
        for(int bdim : blocksizes){
          ++total;
          std::vector<float> h(std::max(n,0));
          fill_pattern(h, pat);
          float ref = (float)cpu_sum(h);

          // Guarded input
          float* d_in_all=nullptr;
          check(cudaMalloc(&d_in_all, (h.size()+2*GUARD)*sizeof(float)), "malloc in");
          std::vector<float> h_in_guard(h.size()+2*GUARD, SENT);
          if(n>0) std::copy(h.begin(), h.end(), h_in_guard.begin()+GUARD);
          check(cudaMemcpy(d_in_all, h_in_guard.data(), (h.size()+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D in");
          float* d_in = d_in_all + GUARD;

          // Guarded output (only one float interior)
          float* d_out_all=nullptr;
          check(cudaMalloc(&d_out_all, (1+2*GUARD)*sizeof(float)), "malloc out");
          std::vector<float> h_out_guard(1+2*GUARD, SENT);
          h_out_guard[GUARD] = 0.0f;
          check(cudaMemcpy(d_out_all, h_out_guard.data(), (1+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D out");
          float* d_out = d_out_all + GUARD;

          // Launch
          dim3 block(bdim), grid(pick_grid(n, bdim));
          size_t shmem = bdim * sizeof(float);
          reduce_sum_arbitrary<<<grid, block, shmem>>>(d_in, d_out, n);
          check(cudaGetLastError(), "launch");
          check(cudaDeviceSynchronize(), "sync");

          // Download
          check(cudaMemcpy(h_out_guard.data(), d_out_all, (1+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
          check(cudaMemcpy(h_in_guard.data(),  d_in_all,  (h.size()+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H in");

          float got = h_out_guard[GUARD];
          bool ok = (n==0) ? (got==0.0f) : almost_equal(got, ref, 1e-3f);

          auto guard_ok=[&](const std::vector<float>& g){
              for(size_t i=0;i<GUARD;i++){ if(g[i]!=SENT||g[g.size()-1-i]!=SENT) return false; }
              return true;
          };
          ok = ok && guard_ok(h_out_guard);
          // input immutability
          if(n>0){
              std::vector<float> in_interior(n);
              std::copy(h_in_guard.begin()+GUARD, h_in_guard.begin()+GUARD+n, in_interior.begin());
              for(int i=0;i<n;i++){
                  if(!almost_equal(in_interior[i], h[i], 1e-6f)){ ok=false; break; }
              }
          }
          ok = ok && guard_ok(h_in_guard);

          printf("n=%-6d pat=%d bdim=%-3d grid=%-3u -> %s (got=%.6f, ref=%.6f)\n", n, pat, bdim, pick_grid(n,bdim), ok?"OK":"FAIL", got, ref);
          if(ok) ++passed;

          cudaFree(d_in_all);
          cudaFree(d_out_all);
        }
      }
    }
    printf("Summary: %d/%d passed\n", passed, total);
    return (passed==total)?0:1;
}