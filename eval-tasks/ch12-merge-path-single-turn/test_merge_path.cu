// test_merge_path.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <climits>

extern "C" __global__
void merge_path_kernel(const int* A, int nA,
                       const int* B, int nB,
                       int* C);

// --- Utilities ---
static void check(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}
static bool equal_vec(const std::vector<int>& a, const std::vector<int>& b){
    return a.size()==b.size() && std::equal(a.begin(), a.end(), b.begin());
}
static void cpu_merge_stable(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C){
    C.resize(A.size()+B.size());
    size_t i=0,j=0,k=0;
    while(i<A.size() || j<B.size()){
        int a = (i<A.size()) ? A[i] : INT_MAX;
        int b = (j<B.size()) ? B[j] : INT_MAX;
        if (a <= b) { C[k++]=a; ++i; }  // stable tie: A first
        else        { C[k++]=b; ++j; }
    }
}
static void fill_sorted(std::vector<int>& v, int pat){
    // fill then sort to guarantee monotonic non-decreasing sequences
    std::mt19937 rng(1234 + pat);
    switch(pat%5){
        case 0: for(size_t i=0;i<v.size();++i) v[i]= (int)i - 50; break;
        case 1: for(size_t i=0;i<v.size();++i) v[i]= (int)((i*13)%97) - 30; break;
        case 2: for(size_t i=0;i<v.size();++i) v[i]= (int)(i%7) - 3; break; // many dups
        case 3: for(size_t i=0;i<v.size();++i) v[i]= (int)((i%2)? -i : i); break;
        default:{
            std::uniform_int_distribution<int> dist(-100,100);
            for(size_t i=0;i<v.size();++i) v[i]=dist(rng);
        } break;
    }
    std::stable_sort(v.begin(), v.end());
}

// --- Test ---
int main(){
    printf("merge-path-single-turn tests\n");

    struct Case { int nA, nB, steps; const char* name; };
    const Case cases[] = {
        {0,0,0,"A=0 B=0"},
        {0,1,0,"A=0 B=1"},
        {1,0,0,"A=1 B=0"},
        {1,1,0,"A=1 B=1"},
        {5,7,0,"A=5 B=7"},
        {32,32,0,"A=32 B=32"},
        {1,1000,0,"A=1 B=1000"},
        {1000,1,0,"A=1000 B=1"},
        {513,257,0,"A=513 B=257"},
        {4096,4096,0,"A=4096 B=4096"},
    };

    const dim3 blocksizes[] = { dim3(128), dim3(256), dim3(512) };

    const size_t GUARD = 4096;
    const int SENT = 13371337;

    int total=0, passed=0;

    for(const auto& cs : cases){
      for(int pat=0; pat<5; ++pat){
        for(auto bdim : blocksizes){
          ++total;

          // Host inputs
          std::vector<int> A(std::max(cs.nA,0)), B(std::max(cs.nB,0));
          fill_sorted(A, pat);
          fill_sorted(B, pat+7);
          std::vector<int> C_ref; cpu_merge_stable(A,B,C_ref);
          const int nA = (int)A.size(), nB=(int)B.size(), nC=nA+nB;

          // Guarded buffers
          int *d_A_all=nullptr, *d_B_all=nullptr, *d_C_all=nullptr;
          check(cudaMalloc(&d_A_all, (nA+2*GUARD)*sizeof(int)), "malloc A");
          check(cudaMalloc(&d_B_all, (nB+2*GUARD)*sizeof(int)), "malloc B");
          check(cudaMalloc(&d_C_all, (nC+2*GUARD)*sizeof(int)), "malloc C");

          std::vector<int> hA_guard(nA+2*GUARD, SENT);
          std::vector<int> hB_guard(nB+2*GUARD, SENT);
          std::vector<int> hC_guard(nC+2*GUARD, SENT);
          if(nA>0) std::copy(A.begin(), A.end(), hA_guard.begin()+GUARD);
          if(nB>0) std::copy(B.begin(), B.end(), hB_guard.begin()+GUARD);

          check(cudaMemcpy(d_A_all, hA_guard.data(), (nA+2*GUARD)*sizeof(int), cudaMemcpyHostToDevice), "H2D A");
          check(cudaMemcpy(d_B_all, hB_guard.data(), (nB+2*GUARD)*sizeof(int), cudaMemcpyHostToDevice), "H2D B");
          check(cudaMemcpy(d_C_all, hC_guard.data(), (nC+2*GUARD)*sizeof(int), cudaMemcpyHostToDevice), "H2D C");

          int* d_A = d_A_all + GUARD;
          int* d_B = d_B_all + GUARD;
          int* d_C = d_C_all + GUARD;

          // Launch config
          auto ceil_div = [](int a,int b){ return (a + b - 1)/b; };
          // For simplicity, use enough threads to cover all data with ~64 elements per thread
          int elements_per_thread = 64;
          int P = std::max(1, nC == 0 ? 1 : ceil_div(nC, elements_per_thread));
          dim3 grid(ceil_div(P, (int)bdim.x));

          merge_path_kernel<<<grid, bdim>>>(d_A, nA, d_B, nB, d_C);
          check(cudaGetLastError(), "launch");
          check(cudaDeviceSynchronize(), "sync");

          // Download
          check(cudaMemcpy(hC_guard.data(), d_C_all, (nC+2*GUARD)*sizeof(int), cudaMemcpyDeviceToHost), "D2H C");
          check(cudaMemcpy(hA_guard.data(), d_A_all, (nA+2*GUARD)*sizeof(int), cudaMemcpyDeviceToHost), "D2H A");
          check(cudaMemcpy(hB_guard.data(), d_B_all, (nB+2*GUARD)*sizeof(int), cudaMemcpyDeviceToHost), "D2H B");

          // Validate
          bool ok = true;
          // Output equals reference
          std::vector<int> C_got(nC);
          if(nC>0) std::copy(hC_guard.begin()+GUARD, hC_guard.begin()+GUARD+nC, C_got.begin());
          ok = ok && equal_vec(C_got, C_ref);

          // Guards unchanged (detect OOB)
          auto guard_ok=[&](const std::vector<int>& g){
              for(size_t i=0;i<GUARD;i++){
                  if(g[i]!=SENT || g[g.size()-1-i]!=SENT) return false;
              } return true;
          };
          ok = ok && guard_ok(hC_guard);

          // Inputs unchanged (interior + guards)
          if(nA>0){
              std::vector<int> A_in(nA);
              std::copy(hA_guard.begin()+GUARD, hA_guard.begin()+GUARD+nA, A_in.begin());
              ok = ok && equal_vec(A_in, A);
          }
          if(nB>0){
              std::vector<int> B_in(nB);
              std::copy(hB_guard.begin()+GUARD, hB_guard.begin()+GUARD+nB, B_in.begin());
              ok = ok && equal_vec(B_in, B);
          }
          ok = ok && guard_ok(hA_guard) && guard_ok(hB_guard);

          printf("Case %-16s pat=%d block=%3u grid=%3u  -> %s\n",
                 cs.name, pat, bdim.x, grid.x, ok?"OK":"FAIL");
          if(ok) ++passed;

          cudaFree(d_A_all);
          cudaFree(d_B_all);
          cudaFree(d_C_all);
        }
      }
    }

    printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}