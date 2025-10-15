// test_sum_2048.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

extern "C" __global__ void reduce_sum_2048(const float* in, float* out);

static void check(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}
static bool almost_equal(float a, float b, float eps=1e-4f){
    float diff = std::fabs(a-b);
    return diff <= eps || diff <= eps * std::max(1.f, std::max(std::fabs(a), std::fabs(b)));
}
static double cpu_sum(const std::vector<float>& v){
    long double s=0;
    for(float x: v) s += x;
    return (double)s;
}
static void fill_pattern(std::vector<float>& v, int pat){
    for (size_t i=0;i<v.size();++i){
        switch(pat%6){
            case 0: v[i] = 1.0f; break;
            case 1: v[i] = float(i%97)*0.25f; break;
            case 2: v[i] = (i%2==0? +3.0f : -3.0f); break;
            case 3: v[i] = 0.001f*float(i) - 0.5f; break;
            case 4: v[i] = (float)((13*i)%113) - 50.0f; break;
            default:v[i] = std::sin(0.01f*float(i)); break;
        }
    }
}

int main(){
    printf("reduction-sum-2048 tests\n");
    const int N = 2048;
    const size_t GUARD = 4096;
    const float SENT = 1337.0f;

    int total=0, passed=0;
    for(int pat=0; pat<6; ++pat){
        ++total;
        // Host input
        std::vector<float> h(N);
        fill_pattern(h, pat);
        double refd = cpu_sum(h);
        float  ref  = (float)refd;

        // Guarded device buffers
        float *d_in_all=nullptr, *d_out_all=nullptr;
        check(cudaMalloc(&d_in_all,  (N+2*GUARD)*sizeof(float)), "malloc in");
        check(cudaMalloc(&d_out_all, (1+2*GUARD)*sizeof(float)), "malloc out");

        std::vector<float> h_in_guard(N+2*GUARD, SENT);
        std::copy(h.begin(), h.end(), h_in_guard.begin()+GUARD);
        std::vector<float> h_out_guard(1+2*GUARD, SENT);
        h_out_guard[GUARD+0] = 0.0f; // interior slot

        check(cudaMemcpy(d_in_all,  h_in_guard.data(),  (N+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D in");
        check(cudaMemcpy(d_out_all, h_out_guard.data(), (1+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D out");

        float* d_in  = d_in_all  + GUARD;
        float* d_out = d_out_all + GUARD;

        // Launch
        dim3 grid(1), block(1024);
        reduce_sum_2048<<<grid, block>>>(d_in, d_out);
        check(cudaGetLastError(), "launch");
        check(cudaDeviceSynchronize(), "sync");

        // Download
        check(cudaMemcpy(h_out_guard.data(), d_out_all, (1+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
        check(cudaMemcpy(h_in_guard.data(),  d_in_all,  (N+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H in");

        // Validate output value
        float got = h_out_guard[GUARD+0];
        bool ok = almost_equal(got, ref, 1e-3f);

        // Validate out guards unchanged
        auto guard_ok = [&](const std::vector<float>& g){
            for(size_t i=0;i<GUARD;i++){ if(g[i]!=SENT || g[g.size()-1-i]!=SENT) return false; }
            return true;
        };
        ok = ok && guard_ok(h_out_guard);

        // Validate input unchanged (interior + guards)
        std::vector<float> h_in_interior(N);
        std::copy(h_in_guard.begin()+GUARD, h_in_guard.begin()+GUARD+N, h_in_interior.begin());
        ok = ok && std::equal(h.begin(), h.end(), h_in_interior.begin(), [](float a,float b){return almost_equal(a,b,1e-6f);});
        ok = ok && guard_ok(h_in_guard);

        printf("pattern %d -> %s (got=%.6f, ref=%.6f)\n", pat, ok?"OK":"FAIL", got, ref);
        if(ok) ++passed;

        cudaFree(d_in_all);
        cudaFree(d_out_all);
    }
    printf("Summary: %d/%d passed\n", passed, total);
    return (passed==total)?0:1;
}