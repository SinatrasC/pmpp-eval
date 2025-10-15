// test_max_arbitrary.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>

extern "C" __global__
void reduce_max_arbitrary(const float* in, float* out, int n);

static void check(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static float cpu_max(const std::vector<float>& v){
    if(v.empty()) return -INFINITY;
    float m = -INFINITY;
    for(float x: v) if (x > m) m = x;
    return m;
}
static void fill_pattern(std::vector<float>& v, int pat){
    for(size_t i=0;i<v.size();++i){
        switch(pat%4){
            case 0: v[i] = (float)((13*i)%503) * 0.01f - 2.0f; break; // mix neg/pos
            case 1: v[i] = -100.0f + 0.5f*(float)(i%100); break;      // mostly negative
            case 2: v[i] = (i%2? -1.0f : +1.0f) * (float)((i%37)+1); break;
            default:v[i] = 0.001f*(float)i - 1.0f; break;
        }
    }
}

static int pick_grid(int n, int b){
    if(n<=0) return 1;
    int blocks = (n + (2*b) - 1) / (2*b);
    return std::min(std::max(blocks,1), 120);
}

int main(){
    printf("reduction-max-arbitrary tests\n");
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
          float ref = cpu_max(h);

          // Guarded input
          float* d_in_all=nullptr;
          check(cudaMalloc(&d_in_all, (h.size()+2*GUARD)*sizeof(float)), "malloc in");
          std::vector<float> h_in_guard(h.size()+2*GUARD, SENT);
          if(n>0) std::copy(h.begin(), h.end(), h_in_guard.begin()+GUARD);
          check(cudaMemcpy(d_in_all, h_in_guard.data(), (h.size()+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D in");
          float* d_in = d_in_all + GUARD;

          // Guarded output (interior = -INF)
          float* d_out_all=nullptr;
          check(cudaMalloc(&d_out_all, (1+2*GUARD)*sizeof(float)), "malloc out");
          std::vector<float> h_out_guard(1+2*GUARD, SENT);
          h_out_guard[GUARD] = -INFINITY;
          check(cudaMemcpy(d_out_all, h_out_guard.data(), (1+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D out");
          float* d_out = d_out_all + GUARD;

          // Launch
          dim3 block(bdim), grid(pick_grid(n, bdim));
          size_t shmem = bdim * sizeof(float);
          reduce_max_arbitrary<<<grid, block, shmem>>>(d_in, d_out, n);
          check(cudaGetLastError(), "launch");
          check(cudaDeviceSynchronize(), "sync");

          // Download
          check(cudaMemcpy(h_out_guard.data(), d_out_all, (1+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
          check(cudaMemcpy(h_in_guard.data(),  d_in_all,  (h.size()+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H in");

          float got = h_out_guard[GUARD];
          bool ok = (n==0) ? (got == -INFINITY) : (got == ref);

          auto guard_ok=[&](const std::vector<float>& g){
              for(size_t i=0;i<GUARD;i++){ if(g[i]!=SENT||g[g.size()-1-i]!=SENT) return false; }
              return true;
          };
          ok = ok && guard_ok(h_out_guard);
          // input unchanged
          if(n>0){
              std::vector<float> in_interior(n);
              std::copy(h_in_guard.begin()+GUARD, h_in_guard.begin()+GUARD+n, in_interior.begin());
              ok = ok && std::equal(h.begin(), h.end(), in_interior.begin());
          }
          ok = ok && guard_ok(h_in_guard);

          printf("n=%-6d pat=%d bdim=%-3d grid=%-3u -> %s (got=%.6f, ref=%.6f)\n",
                 n, pat, bdim, pick_grid(n,bdim), ok?"OK":"FAIL", got, ref);
          if(ok) ++passed;

          cudaFree(d_in_all);
          cudaFree(d_out_all);
        }
      }
    }
    printf("Summary: %d/%d passed\n", passed, total);
    return (passed==total)?0:1;
}