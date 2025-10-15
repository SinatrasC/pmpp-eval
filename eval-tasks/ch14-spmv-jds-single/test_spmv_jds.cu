// ch14-spmv-jds-single / test_spmv_jds.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" void spmv_jds(const int*, const float*, const int*, const int*, const float*, float*, int, int);

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

// Build CSR with patterns, then convert to JDS
struct CSR { int m,n,nnz; std::vector<int> rowPtr, col; std::vector<float> val; };
struct JDS { int m,n,maxJ; std::vector<int> colJds,permute,jdPtr; std::vector<float> valJds; };

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

static JDS csr_to_jds(const CSR& A){
    JDS J{A.m,A.n,0};
    if(A.m==0) {
        J.jdPtr.assign(1, 0); // ensure jdPtr has at least one element
        return J;
    }

    // 1) Compute row lengths & sort by descending length
    std::vector<int> rowLen(A.m);
    for(int i=0;i<A.m;i++) rowLen[i] = A.rowPtr[i+1] - A.rowPtr[i];

    std::vector<int> perm(A.m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return rowLen[a] > rowLen[b]; // descending by length
    });

    J.permute = perm;
    J.maxJ = (A.m>0 && perm.size()>0) ? rowLen[perm[0]] : 0;

    // 2) Build jagged diagonals
    J.jdPtr.assign(J.maxJ+1, 0);
    for(int d=0; d<J.maxJ; d++){
        int count = 0;
        for(int i=0; i<A.m; i++){
            if(rowLen[perm[i]] > d) count++;
        }
        J.jdPtr[d+1] = J.jdPtr[d] + count;
    }

    int totalJds = J.jdPtr[J.maxJ];
    J.colJds.resize(totalJds);
    J.valJds.resize(totalJds);

    for(int d=0; d<J.maxJ; d++){
        int diagBase = J.jdPtr[d];
        int idx = 0;
        for(int i=0; i<A.m; i++){
            int origRow = perm[i];
            if(rowLen[origRow] > d){
                int csrIdx = A.rowPtr[origRow] + d;
                J.colJds[diagBase + idx] = A.col[csrIdx];
                J.valJds[diagBase + idx] = A.val[csrIdx];
                idx++;
            }
        }
    }

    return J;
}

int main(){
    printf("ch14-spmv-jds-single tests\n");

    struct Case{ int m,n,nnz; const char* name; };
    const Case cases[] = {
        {0,0,0, "empty"},
        {1,1,3, "tiny 1x1"},
        {8,8,16,"uniform rows"},
        {64,64,512,"square dense"},
        {64,64,256,"square sparse"},
        {257,129,4096,"rect prime"},
        {128,256,1024,"wide sparse"},
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
          JDS J = csr_to_jds(A);

          std::vector<float> x(cs.n,0.f);
          for(int j=0;j<cs.n;j++) x[j] = 0.01f*(j+1);

          // CPU oracle from CSR
          std::vector<float> y_ref;
          spmv_csr_cpu(A.rowPtr, A.col, A.val, x, y_ref);

          // Guarded device buffers
          int *d_colJ_all=nullptr,*d_perm_all=nullptr,*d_jdPtr_all=nullptr;
          float *d_valJ_all=nullptr,*d_x_all=nullptr,*d_y_all=nullptr;

          int totalJds = J.jdPtr[J.maxJ];
          ck(cudaMalloc(&d_colJ_all, (totalJds + 2*GI)*sizeof(int)), "colJds");
          ck(cudaMalloc(&d_valJ_all, (totalJds + 2*GF)*sizeof(float)), "valJds");
          ck(cudaMalloc(&d_perm_all, (J.m + 2*GI)*sizeof(int)), "permute");
          ck(cudaMalloc(&d_jdPtr_all, (J.maxJ+1 + 2*GI)*sizeof(int)), "jdPtr");
          ck(cudaMalloc(&d_x_all, (J.n + 2*GF)*sizeof(float)), "x");
          ck(cudaMalloc(&d_y_all, (J.m + 2*GF)*sizeof(float)), "y");

          std::vector<int>   h_colJ(totalJds + 2*GI, SI);
          std::vector<float> h_valJ(totalJds + 2*GF, SF);
          std::vector<int>   h_perm(J.m + 2*GI, SI);
          std::vector<int>   h_jdPtr(J.maxJ+1 + 2*GI, SI);
          std::vector<float> h_x(J.n + 2*GF, SF);
          std::vector<float> h_y(J.m + 2*GF, SF);

          if(totalJds>0){
              std::copy(J.colJds.begin(),J.colJds.end(), h_colJ.begin()+GI);
              std::copy(J.valJds.begin(),J.valJds.end(), h_valJ.begin()+GF);
          }
          if(J.m>0) std::copy(J.permute.begin(),J.permute.end(), h_perm.begin()+GI);
          std::copy(J.jdPtr.begin(),J.jdPtr.end(), h_jdPtr.begin()+GI);
          if(J.n>0) std::copy(x.begin(),x.end(), h_x.begin()+GF);

          // **Spec check**: y is zero-initialized
          if(J.m>0) std::fill(h_y.begin()+GF, h_y.begin()+GF+J.m, 0.f);

          ck(cudaMemcpy(d_colJ_all, h_colJ.data(), h_colJ.size()*sizeof(int), cudaMemcpyHostToDevice),"H2D colJ");
          ck(cudaMemcpy(d_valJ_all, h_valJ.data(), h_valJ.size()*sizeof(float), cudaMemcpyHostToDevice),"H2D valJ");
          ck(cudaMemcpy(d_perm_all, h_perm.data(), h_perm.size()*sizeof(int), cudaMemcpyHostToDevice),"H2D perm");
          ck(cudaMemcpy(d_jdPtr_all, h_jdPtr.data(), h_jdPtr.size()*sizeof(int), cudaMemcpyHostToDevice),"H2D jdPtr");
          ck(cudaMemcpy(d_x_all, h_x.data(), h_x.size()*sizeof(float), cudaMemcpyHostToDevice),"H2D x");
          ck(cudaMemcpy(d_y_all, h_y.data(), h_y.size()*sizeof(float), cudaMemcpyHostToDevice),"H2D y");

          int* d_colJds = d_colJ_all + GI;
          float* d_valJds = d_valJ_all + GF;
          int* d_permute = d_perm_all + GI;
          int* d_jdPtr = d_jdPtr_all + GI;
          float* d_x = d_x_all + GF;
          float* d_y = d_y_all + GF;

          // Launch wrapper
          spmv_jds(d_colJds, d_valJds, d_permute, d_jdPtr, d_x, d_y, J.m, J.maxJ);

          // Download & validate
          ck(cudaMemcpy(h_y.data(), d_y_all, h_y.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H y");
          bool ok=true;

          // y equals CPU CSR oracle
          std::vector<float> y_got(J.m,0.f);
          if(J.m>0) std::copy(h_y.begin()+GF, h_y.begin()+GF+J.m, y_got.begin());
          ok = ok && vec_close(y_got, y_ref);

          // Guard canaries & input immutability
          auto guard_ok_i=[&](const std::vector<int>& g){ for(size_t i=0;i<GI;i++) if(g[i]!=SI || g[g.size()-1-i]!=SI) return false; return true; };
          auto guard_ok_f=[&](const std::vector<float>& g){ for(size_t i=0;i<GF;i++) if(g[i]!=SF || g[g.size()-1-i]!=SF) return false; return true; };
          ok = ok && guard_ok_i(h_colJ) && guard_ok_f(h_valJ) && guard_ok_i(h_perm) && guard_ok_i(h_jdPtr) && guard_ok_f(h_x);
          // y guards unchanged
          ok = ok && guard_ok_f(h_y);

          std::printf("JDS m=%4d n=%4d nnz=%5d maxJ=%2d pat=%d block=%3u -> %s\n",
                      A.m,A.n,A.nnz,J.maxJ,pat,bdim.x, ok?"OK":"FAIL");
          if(ok) ++passed;

          cudaFree(d_colJ_all); cudaFree(d_valJ_all); cudaFree(d_perm_all);
          cudaFree(d_jdPtr_all); cudaFree(d_x_all); cudaFree(d_y_all);
        }
      }
    }
    std::printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}