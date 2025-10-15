// ch14-spmv-hyb-single / test_spmv_hyb.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" void spmv_hyb(const int*, const float*, int, int,
                         const int*, const int*, const float*, int,
                         const float*, float*);

static void ck(cudaError_t e,const char* m){
    if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2);}
}
static inline bool feq(float a,float b,float eps=1e-6f){
    float d=std::fabs(a-b); if(d<=eps) return true;
    float ma=std::max(1.f,std::max(std::fabs(a),std::fabs(b))); return d<=eps*ma;
}
static bool vec_close(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-6f){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if(!feq(a[i],b[i],eps)) return false;
    return true;
}

// CPU CSR oracle
static void spmv_csr_cpu(const std::vector<int>& rowPtr,
                         const std::vector<int>& colIdx,
                         const std::vector<float>& val,
                         const std::vector<float>& x,
                         std::vector<float>& y)
{
    int m = (int)rowPtr.size()-1;
    y.assign(m,0.f);
    for(int i=0;i<m;i++){
        float s=0.f;
        for(int k=rowPtr[i]; k<rowPtr[i+1]; ++k) s += val[k] * x[colIdx[k]];
        y[i]=s;
    }
}

// Build CSR with patterns, then split to HYB by first K per row.
struct CSR { int m,n,nnz; std::vector<int> rowPtr, col; std::vector<float> val; };
struct HYB { int m,n,K,nnzC; std::vector<int> colEll,rowCoo,colCoo; std::vector<float> valEll,valCoo; };

static CSR gen_csr(int m,int n,int nnz,int pat){
    CSR A{m,n,0};
    A.rowPtr.assign(m+1,0);
    if(m==0 || n==0 || nnz==0){ A.nnz=0; return A; }

    std::mt19937 rng(2025 + pat*911 + m*7 + n*13 + nnz*3);
    std::uniform_int_distribution<int> rdist(0,m-1), cdist(0,n-1);
    std::uniform_real_distribution<float> fdist(-1.f,1.f);

    std::vector<int> row(nnz), col(nnz);
    std::vector<float> val(nnz);
    for(int k=0;k<nnz;k++){
        int r=0,c=0; float a=fdist(rng);
        switch(pat%6){
            case 0: r=rdist(rng); c=cdist(rng); break;                            // random
            case 1: r=k%m;        c=(k*13 + 7) % n; a = 1.f; break;               // uniform-ish
            case 2: r=(k%2? 0 : m-1); c=cdist(rng); break;                        // tails at two rows
            case 3: r=rdist(rng)%std::max(1,m/4+1); c=cdist(rng); a=1.f; break;   // skew to top rows
            case 4: r=rdist(rng); c=(k%3? 0 : (n-1)); break;                      // column corners
            default: r=rdist(rng); c=cdist(rng); break;
        }
        row[k]= (m? r:0); col[k]= (n? c:0); val[k]= (m==0||n==0)? 0.f: a;
    }
    // bucket per row (stable)
    std::vector<std::vector<int>> rows(m), cols(m);
    std::vector<std::vector<float>> vals(m);
    for(int k=0;k<nnz;k++){ rows[row[k]].push_back(row[k]); cols[row[k]].push_back(col[k]); vals[row[k]].push_back(val[k]); }
    A.rowPtr[0]=0;
    for(int i=0;i<m;i++){ A.rowPtr[i+1] = A.rowPtr[i] + (int)rows[i].size(); }
    A.nnz = A.rowPtr[m];
    A.col.resize(A.nnz); A.val.resize(A.nnz);
    for(int i=0;i<m;i++){
        int base=A.rowPtr[i];
        for(size_t t=0;t<rows[i].size();++t){
            A.col[base+(int)t]=cols[i][t];
            A.val[base+(int)t]=vals[i][t];
        }
    }
    return A;
}

static HYB csr_to_hyb(const CSR& A,int K){
    HYB H{A.m,A.n,K,0};
    H.colEll.assign((size_t)A.m*K, -1);
    H.valEll.assign((size_t)A.m*K, 0.f);
    for(int i=0;i<A.m;i++){
        int len = A.rowPtr[i+1]-A.rowPtr[i];
        int take = std::min(K,len);
        int base_ell = i*K;
        // first K -> ELL
        for(int t=0;t<take;t++){
            int k = A.rowPtr[i]+t;
            H.colEll[base_ell+t] = A.col[k];
            H.valEll[base_ell+t] = A.val[k];
        }
        // tail -> COO
        for(int t=take;t<len;t++){
            int k = A.rowPtr[i]+t;
            H.rowCoo.push_back(i);
            H.colCoo.push_back(A.col[k]);
            H.valCoo.push_back(A.val[k]);
        }
    }
    H.nnzC = (int)H.rowCoo.size();
    return H;
}

int main(){
    printf("ch14-spmv-hyb-single tests\n");

    struct Case{ int m,n,nnz,K; const char* name; };
    const Case cases[] = {
        {0,0,0, 0, "empty"},
        {1,1,3, 2, "tiny 1x1 K2 tails"},
        {8,8,16,2, "uniform rows (<=K)"},
        {64,64,512,16,"pure-ELL (<=K)"},
        {64,64,512, 0,"tails-only (pure COO)"},
        {64,64,1024,4,"skewed rows (long tails)"},
        {257,129,4096,8,"mixed prime"},
    };
    const dim3 blocks[] = { dim3(128), dim3(256), dim3(512) };

    const size_t GI=1024; const int SI=0x7f7f7f7f;
    const size_t GF=1024; const float SF=1337.f;

    int total=0, passed=0;
    for(const auto& cs: cases){
      for(int pat=0; pat<6; ++pat){
        for(auto bdim: blocks){
          ++total;

          CSR A = gen_csr(cs.m, cs.n, cs.nnz, pat);
          HYB H = csr_to_hyb(A, cs.K);

          std::vector<float> x(cs.n,0.f);
          for(int j=0;j<cs.n;j++) x[j] = 0.01f*(j+1);

          // CPU oracle from CSR
          std::vector<float> y_ref;
          spmv_csr_cpu(A.rowPtr, A.col, A.val, x, y_ref);

          // Guarded device buffers
          int *d_colE_all=nullptr,*d_rowC_all=nullptr,*d_colC_all=nullptr;
          float *d_valE_all=nullptr,*d_valC_all=nullptr,*d_x_all=nullptr,*d_y_all=nullptr;

          ck(cudaMalloc(&d_colE_all, (H.m*H.K + 2*GI)*sizeof(int)), "colEll");
          ck(cudaMalloc(&d_valE_all, (H.m*H.K + 2*GF)*sizeof(float)), "valEll");
          ck(cudaMalloc(&d_rowC_all, (H.nnzC   + 2*GI)*sizeof(int)), "rowC");
          ck(cudaMalloc(&d_colC_all, (H.nnzC   + 2*GI)*sizeof(int)), "colC");
          ck(cudaMalloc(&d_valC_all, (H.nnzC   + 2*GF)*sizeof(float)), "valC");
          ck(cudaMalloc(&d_x_all,    (H.n      + 2*GF)*sizeof(float)), "x");
          ck(cudaMalloc(&d_y_all,    (H.m      + 2*GF)*sizeof(float)), "y");

          std::vector<int>   h_colE(H.m*H.K + 2*GI, SI);
          std::vector<float> h_valE(H.m*H.K + 2*GF, SF);
          std::vector<int>   h_rowC(H.nnzC + 2*GI, SI);
          std::vector<int>   h_colC(H.nnzC + 2*GI, SI);
          std::vector<float> h_valC(H.nnzC + 2*GF, SF);
          std::vector<float> h_x(   H.n    + 2*GF, SF);
          std::vector<float> h_y(   H.m    + 2*GF, SF);

          if(H.m*H.K>0){
              std::copy(H.colEll.begin(),H.colEll.end(), h_colE.begin()+GI);
              std::copy(H.valEll.begin(),H.valEll.end(), h_valE.begin()+GF);
          }
          if(H.nnzC>0){
              std::copy(H.rowCoo.begin(),H.rowCoo.end(), h_rowC.begin()+GI);
              std::copy(H.colCoo.begin(),H.colCoo.end(), h_colC.begin()+GI);
              std::copy(H.valCoo.begin(),H.valCoo.end(), h_valC.begin()+GF);
          }
          if(H.n>0) std::copy(x.begin(),x.end(), h_x.begin()+GF);

          // **Spec check**: y is zero-initialized (then only modified by kernels)
          if(H.m>0) std::fill(h_y.begin()+GF, h_y.begin()+GF+H.m, 0.f);

          ck(cudaMemcpy(d_colE_all, h_colE.data(), h_colE.size()*sizeof(int),   cudaMemcpyHostToDevice),"H2D colEll");
          ck(cudaMemcpy(d_valE_all, h_valE.data(), h_valE.size()*sizeof(float), cudaMemcpyHostToDevice),"H2D valEll");
          ck(cudaMemcpy(d_rowC_all, h_rowC.data(), h_rowC.size()*sizeof(int),   cudaMemcpyHostToDevice),"H2D rowC");
          ck(cudaMemcpy(d_colC_all, h_colC.data(), h_colC.size()*sizeof(int),   cudaMemcpyHostToDevice),"H2D colC");
          ck(cudaMemcpy(d_valC_all, h_valC.data(), h_valC.size()*sizeof(float), cudaMemcpyHostToDevice),"H2D valC");
          ck(cudaMemcpy(d_x_all,    h_x.data(),    h_x.size()*sizeof(float),    cudaMemcpyHostToDevice),"H2D x");
          ck(cudaMemcpy(d_y_all,    h_y.data(),    h_y.size()*sizeof(float),    cudaMemcpyHostToDevice),"H2D y");

          int* d_colEll = d_colE_all + GI;
          float* d_valEll = d_valE_all + GF;
          int* d_rowC = d_rowC_all + GI;
          int* d_colC = d_colC_all + GI;
          float* d_valC = d_valC_all + GF;
          float* d_x = d_x_all + GF;
          float* d_y = d_y_all + GF;

          // Launch wrapper
          spmv_hyb(d_colEll, d_valEll, H.m, H.K,
                   d_rowC, d_colC, d_valC, H.nnzC,
                   d_x, d_y);

          // Download & validate
          ck(cudaMemcpy(h_y.data(), d_y_all, h_y.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H y");
          bool ok=true;

          // y equals CPU CSR oracle
          std::vector<float> y_got(H.m,0.f);
          if(H.m>0) std::copy(h_y.begin()+GF, h_y.begin()+GF+H.m, y_got.begin());
          ok = ok && vec_close(y_got, y_ref, 5e-5f);  // Higher tolerance for atomic operations

          // Guard canaries & input immutability
          auto guard_ok_i=[&](const std::vector<int>& g){ for(size_t i=0;i<GI;i++) if(g[i]!=SI || g[g.size()-1-i]!=SI) return false; return true; };
          auto guard_ok_f=[&](const std::vector<float>& g){ for(size_t i=0;i<GF;i++) if(g[i]!=SF || g[g.size()-1-i]!=SF) return false; return true; };
          ok = ok && guard_ok_i(h_colE) && guard_ok_f(h_valE) && guard_ok_i(h_rowC) && guard_ok_i(h_colC) && guard_ok_f(h_valC) && guard_ok_f(h_x);
          // y guards unchanged
          ok = ok && guard_ok_f(h_y);

          std::printf("HYB m=%4d n=%4d nnz=%5d K=%2d pat=%d block=%3u -> %s\n",
                      A.m,A.n,A.nnz,cs.K,pat,bdim.x, ok?"OK":"FAIL");
          if(ok) ++passed;

          cudaFree(d_colE_all); cudaFree(d_valE_all);
          cudaFree(d_rowC_all); cudaFree(d_colC_all); cudaFree(d_valC_all);
          cudaFree(d_x_all);    cudaFree(d_y_all);
        }
      }
    }
    std::printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}