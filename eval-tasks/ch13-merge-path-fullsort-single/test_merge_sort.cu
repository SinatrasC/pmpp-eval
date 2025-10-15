#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <random>
#include <cstdint>
#include <cstdio>

extern "C" void gpu_merge_sort(const uint32_t* d_in, uint32_t* d_out, int n);

static void ck(cudaError_t e,const char*m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static void fill_pattern(std::vector<uint32_t>& v, int pat){
    std::mt19937 rng(42+pat);
    switch(pat%5){
        case 0: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)i; break;
        case 1: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)((i*13)^(i>>1)); break;
        case 2: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)(i%17); break;
        case 3: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)((i&1)? i: ~i); break;
        default:{ std::uniform_int_distribution<uint32_t> d(0u,0xffffffffu); for(size_t i=0;i<v.size();++i) v[i]=d(rng); }
    }
}

int main(){
    printf("ch13-merge-path-fullsort-single\n");
    const int sizes[] = {0,1,5,32,1000,4096,20000};
    int total=0, pass=0;

    for(int n: sizes){
        for(int pat=0; pat<5; ++pat){
            ++total;
            std::vector<uint32_t> h_in(n), h_ref(n), h_out(n);
            fill_pattern(h_in, pat);
            h_ref = h_in;
            std::stable_sort(h_ref.begin(), h_ref.end());

            uint32_t *d_in=nullptr, *d_out=nullptr;
            ck(cudaMalloc(&d_in, n*sizeof(uint32_t)), "malloc in");
            ck(cudaMalloc(&d_out,n*sizeof(uint32_t)), "malloc out");
            ck(cudaMemcpy(d_in, h_in.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D");

            gpu_merge_sort(d_in, d_out, n);
            ck(cudaMemcpy(h_out.data(), d_out, n*sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H");

            bool ok = (h_out == h_ref);
            printf("n=%7d pat=%d -> %s\n", n, pat, ok?"OK":"FAIL");
            if(ok) ++pass;

            cudaFree(d_in); cudaFree(d_out);
        }
    }
    printf("Summary: %d/%d passed\n", pass, total);
    return (pass==total)?0:1;
}