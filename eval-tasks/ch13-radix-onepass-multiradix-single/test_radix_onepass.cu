#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <random>
#include <cstdint>
#include <cstdio>
#include <cassert>

extern "C" void radix_onepass_multiradix(const uint32_t*, uint32_t*, int, int, int);

static void ck(cudaError_t e,const char*m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static void cpu_onepass(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, int r, int shift){
    int n=(int)in.size(); if(n==0){out.clear();return;}
    int B=1<<r, mask=B-1;
    std::vector<uint32_t> cnt(B,0);
    std::vector<uint32_t> digit(n);
    for(int i=0;i<n;i++){ digit[i]=(in[i]>>shift)&mask; cnt[digit[i]]++; }
    std::vector<uint32_t> base(B,0); // exclusive
    for(int b=1;b<B;b++) base[b]=base[b-1]+cnt[b-1];
    out.resize(n);
    std::vector<uint32_t> off=base;
    for(int i=0;i<n;i++){ out[ off[digit[i]]++ ] = in[i]; } // stable
}

static void fill_pattern(std::vector<uint32_t>& v, int pat){
    std::mt19937 rng(1234+pat);
    switch(pat%5){
        case 0: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)i*2654435761u; break;
        case 1: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)((i*13)^(i>>3)); break;
        case 2: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)(i%17); break; // many dups
        case 3: for(size_t i=0;i<v.size();++i) v[i]=(uint32_t)((i&1)? i: ~i); break;
        default:{ std::uniform_int_distribution<uint32_t> d(0u,0xffffffffu); for(size_t i=0;i<v.size();++i) v[i]=d(rng); }
    }
}

int main(){
    printf("ch13-radix-onepass-multiradix-single\n");
    const int sizes[] = {0,1,5,32,1000,4096,20000};
    const int rs[] = {1,2,4};
    const int shifts[] = {0,4,8,16,24};

    int total=0, pass=0;
    for(int n : sizes){
        for(int pat=0; pat<5; ++pat){
            std::vector<uint32_t> h_in(n);
            fill_pattern(h_in, pat);
            uint32_t *d_in=nullptr, *d_out=nullptr;
            ck(cudaMalloc(&d_in, n*sizeof(uint32_t)), "malloc in");
            ck(cudaMalloc(&d_out,n*sizeof(uint32_t)), "malloc out");
            ck(cudaMemcpy(d_in,h_in.data(),n*sizeof(uint32_t),cudaMemcpyHostToDevice),"H2D");

            for(int r: rs){
                for(int sh: shifts){
                    ++total;
                    std::vector<uint32_t> h_ref;
                    cpu_onepass(h_in, h_ref, r, sh);

                    radix_onepass_multiradix(d_in, d_out, n, r, sh);
                    std::vector<uint32_t> h_out(n);
                    ck(cudaMemcpy(h_out.data(), d_out, n*sizeof(uint32_t), cudaMemcpyDeviceToHost),"D2H");

                    bool ok = (h_out==h_ref);
                    printf("n=%7d pat=%d r=%d sh=%2d -> %s\n", n, pat, r, sh, ok?"OK":"FAIL");
                    if(ok) ++pass;
                }
            }

            cudaFree(d_in); cudaFree(d_out);
        }
    }
    printf("Summary: %d/%d passed\n", pass, total);
    return (pass==total)?0:1;
}