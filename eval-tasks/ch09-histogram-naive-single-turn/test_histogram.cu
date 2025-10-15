// test_histogram.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <cmath>

extern __global__ void histogram_kernel(const int* in, unsigned int* hist,
                                        size_t N, int num_bins);

static void check(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}
static inline size_t cdiv(size_t a, size_t b){ return (a+b-1)/b; }

// CPU oracle (64-bit accumulator)
static std::vector<unsigned long long> cpu_hist(const std::vector<int>& in, int num_bins){
    std::vector<unsigned long long> H(num_bins,0ULL);
    for(size_t i=0;i<in.size();++i){
        int b=in[i]; if(b>=0 && b<num_bins) H[b]++; }
    return H;
}
static bool equal_u32_u64(const std::vector<unsigned int>& a,
                          const std::vector<unsigned long long>& b){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if((unsigned long long)a[i]!=b[i]) return false;
    return true;
}

// adversarial fillers
static void fill_pattern(std::vector<int>& v, int num_bins, int pat){
    const size_t N=v.size();
    switch(pat%4){
        case 0: // uniform modulo
            for(size_t i=0;i<N;++i) v[i] = (int)(i% (size_t)std::max(1,num_bins));
            break;
        case 1: // heavy skew to 0
            for(size_t i=0;i<N;++i) v[i] = (i%16)==0 ? 0 : (int)((i+7)%std::max(1,num_bins));
            break;
        case 2: // alternating extremes
            for(size_t i=0;i<N;++i) v[i] = (i&1)? (num_bins-1) : 0;
            break;
        default: // random-ish LCG modulo
            { uint32_t x=1234567u;
              for(size_t i=0;i<N;++i){ x=1664525u*x+1013904223u; v[i]=(int)(x% (uint32_t)std::max(1,num_bins)); } }
            break;
    }
}

int main(){
    printf("ch09 naive histogram — tests\n");

    const size_t GUARD_ELEMS = 1024;
    const int INPUT_SENT = -123456789;
    const unsigned int HIST_SENT = 0xDEADBEEFu;

    const int bin_sets[] = {1,2,7,128,256,1024};
    const size_t sizes[] = {0,1,17,257,4093, (size_t)1<<20};
    const int blocksizes[] = {63,128,256};

    int total=0, passed=0;

    for(int nb : bin_sets){
      for(size_t N : sizes){
        for(int bs : blocksizes){
          for(int pat=0; pat<4; ++pat){

            // host input + guards
            std::vector<int> h_in(N);
            fill_pattern(h_in, nb, pat);
            auto Href = cpu_hist(h_in, nb);

            // guarded device buffers
            int* d_in_base=nullptr;
            unsigned int* d_hist_base=nullptr;
            check(cudaMalloc(&d_in_base, (N+2*GUARD_ELEMS)*sizeof(int)),"malloc in");
            check(cudaMalloc(&d_hist_base,(nb+2*GUARD_ELEMS)*sizeof(unsigned int)),"malloc hist");

            std::vector<int> h_in_guard(N+2*GUARD_ELEMS, INPUT_SENT);
            std::copy(h_in.begin(), h_in.end(), h_in_guard.begin()+GUARD_ELEMS);

            std::vector<unsigned int> h_hist_guard(nb+2*GUARD_ELEMS, HIST_SENT);
            // interior histogram must start at 0
            std::fill(h_hist_guard.begin()+GUARD_ELEMS, h_hist_guard.begin()+GUARD_ELEMS+nb, 0u);

            check(cudaMemcpy(d_in_base, h_in_guard.data(),
                             (N+2*GUARD_ELEMS)*sizeof(int), cudaMemcpyHostToDevice),"upload in");
            check(cudaMemcpy(d_hist_base, h_hist_guard.data(),
                             (nb+2*GUARD_ELEMS)*sizeof(unsigned int), cudaMemcpyHostToDevice),"upload hist");

            int* d_in = d_in_base + GUARD_ELEMS;
            unsigned int* d_hist = d_hist_base + GUARD_ELEMS;

            int blocks = (int)std::min(cdiv(N, (size_t)bs), (size_t)65535);
            if(blocks==0) blocks=1;

            // launch
            histogram_kernel<<<blocks, bs>>>(d_in, d_hist, N, nb);
            check(cudaGetLastError(),"launch");
            check(cudaDeviceSynchronize(),"sync");

            // download
            check(cudaMemcpy(h_in_guard.data(), d_in_base,
                             (N+2*GUARD_ELEMS)*sizeof(int), cudaMemcpyDeviceToHost),"down in");
            check(cudaMemcpy(h_hist_guard.data(), d_hist_base,
                             (nb+2*GUARD_ELEMS)*sizeof(unsigned int), cudaMemcpyDeviceToHost),"down hist");

            // extract interior
            std::vector<unsigned int> Hgot(nb);
            std::copy(h_hist_guard.begin()+GUARD_ELEMS, h_hist_guard.begin()+GUARD_ELEMS+nb, Hgot.begin());

            bool ok=true; ++total;

            // 1) exact match vs CPU oracle
            ok = ok && equal_u32_u64(Hgot, Href);

            // 2) input immutability (interior)
            std::vector<int> h_in_after(N);
            std::copy(h_in_guard.begin()+GUARD_ELEMS, h_in_guard.begin()+GUARD_ELEMS+N, h_in_after.begin());
            ok = ok && std::equal(h_in_after.begin(), h_in_after.end(), h_in.begin());

            // 3) input guards unchanged
            auto guard_ok_in = [&](const std::vector<int>& g){
                for(size_t t=0;t<GUARD_ELEMS;t++){
                    if(g[t]!=INPUT_SENT) return false;
                    if(g[g.size()-1-t]!=INPUT_SENT) return false;
                } return true;
            };
            ok = ok && guard_ok_in(h_in_guard);

            // 4) hist guards unchanged
            auto guard_ok_hist = [&](const std::vector<unsigned int>& g){
                for(size_t t=0;t<GUARD_ELEMS;t++){
                    if(g[t]!=HIST_SENT) return false;
                    if(g[g.size()-1-t]!=HIST_SENT) return false;
                } return true;
            };
            ok = ok && guard_ok_hist(h_hist_guard);

            printf("bins=%4d N=%8zu bs=%3d pat=%d -> %s\n", nb, N, bs, pat, ok?"OK":"FAIL");
            if(ok) ++passed;

            cudaFree(d_in_base);
            cudaFree(d_hist_base);
          }
        }
      }
    }

    printf("Summary: %d / %d tests passed\n", passed, total);
    printf("All tests passed.\n");
    return (passed==total)?0:1;
}