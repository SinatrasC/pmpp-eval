// ch14-coo-to-csr-single / test_coo_to_csr.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" void coo_to_csr(const int* d_row, const int* d_col, const float* d_val,
                           int nnz, int m, int n,
                           int* d_rowPtr, int* d_colCSR, float* d_valCSR);

static void ck(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static inline bool feq(float a,float b,float eps=1e-6f){
    float d = std::fabs(a-b); if (d<=eps) return true;
    float ma = std::max(1.f,std::max(std::fabs(a),std::fabs(b))); return d<=eps*ma;
}
static bool vec_close(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-6f){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if(!feq(a[i],b[i],eps)) return false;
    return true;
}

// CPU stable COO -> CSR
static void coo_to_csr_cpu(const std::vector<int>& row,
                           const std::vector<int>& col,
                           const std::vector<float>& val,
                           int nnz, int m, int /*n*/,
                           std::vector<int>& rowPtr,
                           std::vector<int>& colCSR,
                           std::vector<float>& valCSR)
{
    rowPtr.assign(m+1,0);
    for (int i=0;i<nnz;i++) { int r = row[i]; if(r>=0 && r<m) rowPtr[r+1]++; }
    for (int i=0;i<m;i++) rowPtr[i+1] += rowPtr[i];

    colCSR.resize(nnz); valCSR.resize(nnz);
    if (m > 0) {
        std::vector<int> next(rowPtr.begin(), rowPtr.begin()+m);
        for (int i=0;i<nnz;i++) {
            int r = row[i];
            if (r >= 0 && r < m) {
                int p = next[r]++;       // stable
                colCSR[p] = col[i];
                valCSR[p] = val[i];
            }
        }
    }
}

// CPU CSR SpMV
static void spmv_csr_cpu(const std::vector<int>& rowPtr,
                         const std::vector<int>& col,
                         const std::vector<float>& val,
                         const std::vector<float>& x,
                         std::vector<float>& y)
{
    int m = (int)rowPtr.size()-1; y.assign(m,0.f);
    for (int i=0;i<m;i++) {
        float s=0.f;
        for (int k=rowPtr[i]; k<rowPtr[i+1]; ++k) s += val[k]*x[col[k]];
        y[i]=s;
    }
}

// Patterns for COO generation
struct COO { int m,n,nnz; std::vector<int> row,col; std::vector<float> val; };

static COO gen_coo(int m, int n, int nnz, int pat){
    COO A{m,n,nnz};
    A.row.resize(nnz); A.col.resize(nnz); A.val.resize(nnz);

    std::mt19937 rng(202501 + pat*17 + m*3 + n*5 + nnz*7);
    std::uniform_int_distribution<int> rdist(0, std::max(0,m-1));
    std::uniform_int_distribution<int> cdist(0, std::max(0,n-1));
    std::uniform_real_distribution<float> fdist(-1.f,1.f);

    // base entries
    for (int i=0;i<nnz;i++){
        int r=0,c=0; float a=fdist(rng);
        switch(pat % 6){
            case 0: // random unsorted
                r = rdist(rng); c = cdist(rng); break;
            case 1: // reverse rows order
                r = (nnz-1 - i) % std::max(1,m); c = cdist(rng); break;
            case 2: // duplicates (same (r,c) repeated)
                r = (i % std::max(1,m)); c = (i % std::max(1,n/3+1)); break;
            case 3: // empty rows (heavily biased to first quarter)
                r = (rdist(rng) % std::max(1,m/4+1)); c = cdist(rng); break;
            case 4: // already grouped by row (ascending), but arbitrary cols
            {
                int chunk = std::max(1, nnz/std::max(1,m));
                r = std::min(m-1, i / chunk);
                c = cdist(rng);
            } break;
            default:// many repeats of few cols
                r = rdist(rng);
                c = (i % 5);
                break;
        }
        A.row[i]= (m? r:0);
        A.col[i]= (n? c:0);
        A.val[i]= (m==0||n==0)? 0.f : a;
    }
    return A;
}

int main(){
    printf("ch14-coo-to-csr-single tests\n");

    struct Case { int m,n,nnz; const char* name; };
    const Case cases[] = {
        {0,0,0,"empty"},
        {4,4,0,"no nnz"},
        {4,4,5,"tiny mixed"},
        {32,32,64,"small"},
        {128,77,1024,"rectangular"},
        {513,257,4096,"prime irregular"},
        {1024,1024,10000,"larger"}
    };

    const size_t GI=1024; const int   SI=0x7f7f7f7f;
    const size_t GF=1024; const float SF=1337.f;

    int total=0, passed=0;
    for (const auto& cs : cases){
      for (int pat=0; pat<6; ++pat){
        ++total;

        COO A = gen_coo(cs.m, cs.n, cs.nnz, pat);

        // CPU oracle (stable)
        std::vector<int> rowPtr_ref, col_ref;
        std::vector<float> val_ref;
        coo_to_csr_cpu(A.row, A.col, A.val, A.nnz, A.m, A.n, rowPtr_ref, col_ref, val_ref);

        // x for SpMV check
        std::vector<float> x(cs.n, 0.f);
        for (int j=0;j<cs.n;j++) x[j] = 0.01f*(j+1);
        std::vector<float> y_ref;
        spmv_csr_cpu(rowPtr_ref, col_ref, val_ref, x, y_ref);

        // Guarded device buffers
        int *d_row_all=nullptr, *d_col_all=nullptr, *d_rowPtr_all=nullptr, *d_colCSR_all=nullptr;
        float *d_val_all=nullptr, *d_valCSR_all=nullptr, *d_x_all=nullptr, *d_y_all=nullptr;

        ck(cudaMalloc(&d_row_all,    (A.nnz + 2*GI)*sizeof(int)),   "row");
        ck(cudaMalloc(&d_col_all,    (A.nnz + 2*GI)*sizeof(int)),   "col");
        ck(cudaMalloc(&d_val_all,    (A.nnz + 2*GF)*sizeof(float)), "val");
        ck(cudaMalloc(&d_rowPtr_all, (cs.m+1 + 2*GI)*sizeof(int)),  "rowPtr");
        ck(cudaMalloc(&d_colCSR_all, (A.nnz + 2*GI)*sizeof(int)),   "colCSR");
        ck(cudaMalloc(&d_valCSR_all, (A.nnz + 2*GF)*sizeof(float)), "valCSR");
        ck(cudaMalloc(&d_x_all,      (cs.n + 2*GF)*sizeof(float)),  "x");
        ck(cudaMalloc(&d_y_all,      (cs.m + 2*GF)*sizeof(float)),  "y");

        std::vector<int>   h_row (A.nnz + 2*GI, SI);
        std::vector<int>   h_col (A.nnz + 2*GI, SI);
        std::vector<float> h_val (A.nnz + 2*GF, SF);
        std::vector<int>   h_rowPtr(cs.m+1 + 2*GI, SI);
        std::vector<int>   h_colCSR(A.nnz + 2*GI, SI);
        std::vector<float> h_valCSR(A.nnz + 2*GF, SF);
        std::vector<float> h_x(cs.n + 2*GF, SF);
        std::vector<float> h_y(cs.m + 2*GF, SF);

        if (A.nnz>0){
            std::copy(A.row.begin(), A.row.end(), h_row.begin()+GI);
            std::copy(A.col.begin(), A.col.end(), h_col.begin()+GI);
            std::copy(A.val.begin(), A.val.end(), h_val.begin()+GF);
        }
        if (cs.n>0) std::copy(x.begin(), x.end(), h_x.begin()+GF);
        if (cs.m>0) std::fill(h_y.begin()+GF, h_y.begin()+GF+cs.m, 0.f);

        ck(cudaMemcpy(d_row_all, h_row.data(), h_row.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D row");
        ck(cudaMemcpy(d_col_all, h_col.data(), h_col.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D col");
        ck(cudaMemcpy(d_val_all, h_val.data(), h_val.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D val");
        ck(cudaMemcpy(d_rowPtr_all, h_rowPtr.data(), h_rowPtr.size()*sizeof(int), cudaMemcpyHostToDevice), "H2D rowPtr init");
        ck(cudaMemcpy(d_colCSR_all, h_colCSR.data(), h_colCSR.size()*sizeof(int), cudaMemcpyHostToDevice), "H2D colCSR init");
        ck(cudaMemcpy(d_valCSR_all, h_valCSR.data(), h_valCSR.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D valCSR init");
        ck(cudaMemcpy(d_x_all, h_x.data(), h_x.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D x");
        ck(cudaMemcpy(d_y_all, h_y.data(), h_y.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D y");

        int* d_row    = d_row_all + GI;
        int* d_col    = d_col_all + GI;
        float* d_val  = d_val_all + GF;
        int* d_rowPtr = d_rowPtr_all + GI;
        int* d_colCSR = d_colCSR_all + GI;
        float* d_valCSR = d_valCSR_all + GF;
        const float* d_x = d_x_all + GF;
        float* d_y = d_y_all + GF;

        // Run conversion
        coo_to_csr(d_row, d_col, d_val, A.nnz, cs.m, cs.n, d_rowPtr, d_colCSR, d_valCSR);
        ck(cudaGetLastError(), "coo_to_csr");
        ck(cudaDeviceSynchronize(), "sync");

        // Download results
        ck(cudaMemcpy(h_rowPtr.data(), d_rowPtr_all, h_rowPtr.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H rowPtr");
        ck(cudaMemcpy(h_colCSR.data(), d_colCSR_all, h_colCSR.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H colCSR");
        ck(cudaMemcpy(h_valCSR.data(), d_valCSR_all, h_valCSR.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H valCSR");

        // Validate conversion exactness
        bool ok = true;

        // rowPtr
        std::vector<int> rowPtr_got(cs.m+1, 0);
        if (cs.m>=0) std::copy(h_rowPtr.begin()+GI, h_rowPtr.begin()+GI+cs.m+1, rowPtr_got.begin());
        ok = ok && (rowPtr_got == rowPtr_ref);
        ok = ok && (rowPtr_got.back() == A.nnz);

        // col/val exact
        std::vector<int> col_got(A.nnz,0);
        std::vector<float> val_got(A.nnz,0.f);
        if (A.nnz>0){
            std::copy(h_colCSR.begin()+GI, h_colCSR.begin()+GI+A.nnz, col_got.begin());
            std::copy(h_valCSR.begin()+GF, h_valCSR.begin()+GF+A.nnz, val_got.begin());
        }
        ok = ok && (col_got == col_ref);
        ok = ok && vec_close(val_got, val_ref);

        // OOB canaries / input immutability
        auto guard_ok_i=[&](const std::vector<int>& g){ for(size_t i=0;i<GI;i++) if(g[i]!=SI || g[g.size()-1-i]!=SI) return false; return true; };
        auto guard_ok_f=[&](const std::vector<float>& g){ for(size_t i=0;i<GF;i++) if(g[i]!=SF || g[g.size()-1-i]!=SF) return false; return true; };
        ok = ok && guard_ok_i(h_row) && guard_ok_i(h_col) && guard_ok_f(h_val);
        ok = ok && guard_ok_i(h_rowPtr) && guard_ok_i(h_colCSR) && guard_ok_f(h_valCSR);
        ok = ok && guard_ok_f(h_x) && guard_ok_f(h_y);

        // Secondary check: CSR SpMV equals CPU oracle
        if (cs.m>0 && cs.n>0){
            // quick SpMV on GPU result (do CPU for simplicity)
            std::vector<float> y_got;
            spmv_csr_cpu(rowPtr_got, col_got, val_got, x, y_got);
            ok = ok && vec_close(y_got, y_ref);
        }

        std::printf("COO->CSR m=%4d n=%4d nnz=%6d pat=%d -> %s\n",
                    cs.m, cs.n, cs.nnz, pat, ok?"OK":"FAIL");
        if(ok) ++passed;

        cudaFree(d_row_all); cudaFree(d_col_all); cudaFree(d_val_all);
        cudaFree(d_rowPtr_all); cudaFree(d_colCSR_all); cudaFree(d_valCSR_all);
        cudaFree(d_x_all); cudaFree(d_y_all);
      }
    }
    std::printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}