// test_heat.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

extern __global__ void heat_step_kernel(const float* in, float* out,
                                         unsigned int N, float alpha, float dt, float dx);

// --------------------------------- Utilities ---------------------------------
static inline size_t I(size_t i, size_t j, size_t k, size_t N) {
    return i*N*N + j*N + k;
}

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(2);
    }
}

// CPU oracle for a single step (Dirichlet copy-through boundary)
static void heat_step_cpu(const std::vector<float>& in,
                          std::vector<float>& out,
                          unsigned N, float alpha, float dt, float dx)
{
    auto idx = [&](unsigned i,unsigned j,unsigned k){ return I(i,j,k,N); };

    if (N < 3) {
        out = in;
        return;
    }

    const float r = alpha * dt / (dx * dx);

    // copy boundaries
    for (unsigned i=0;i<N;i++){
        for (unsigned j=0;j<N;j++){
            out[idx(i,j,0)]     = in[idx(i,j,0)];
            out[idx(i,j,N-1)]   = in[idx(i,j,N-1)];
        }
    }
    for (unsigned i=0;i<N;i++){
        for (unsigned k=0;k<N;k++){
            out[idx(i,0,k)]     = in[idx(i,0,k)];
            out[idx(i,N-1,k)]   = in[idx(i,N-1,k)];
        }
    }
    for (unsigned j=0;j<N;j++){
        for (unsigned k=0;k<N;k++){
            out[idx(0,j,k)]     = in[idx(0,j,k)];
            out[idx(N-1,j,k)]   = in[idx(N-1,j,k)];
        }
    }

    // interior update
    for (unsigned i=1;i<N-1;i++){
        for (unsigned j=1;j<N-1;j++){
            for (unsigned k=1;k<N-1;k++){
                float c  = in[idx(i,j,k)];
                float xm = in[idx(i-1,j,k)];
                float xp = in[idx(i+1,j,k)];
                float ym = in[idx(i,j-1,k)];
                float yp = in[idx(i,j+1,k)];
                float zm = in[idx(i,j,k-1)];
                float zp = in[idx(i,j,k+1)];
                out[idx(i,j,k)] = c + r * ((xm+xp+ym+yp+zm+zp) - 6.f*c);
            }
        }
    }
}

static bool almost_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float eps = 1e-4f)
{
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float diff = std::fabs(a[i]-b[i]);
        if (!(diff <= eps || diff <= eps*std::max(1.f,std::max(std::fabs(a[i]), std::fabs(b[i]))))) {
            return false;
        }
    }
    return true;
}

// Fill patterns (adversarial)
static void fill_pattern(std::vector<float>& v, unsigned N, int pat) {
    for (unsigned i=0;i<N;i++){
        for (unsigned j=0;j<N;j++){
            for (unsigned k=0;k<N;k++){
                size_t p = I(i,j,k,N);
                switch (pat % 4) {
                    case 0: v[p] = float(i + j + k); break;
                    case 1: v[p] = float((13*i + 7*j + 3*k) % 101) * 0.5f; break;
                    case 2: v[p] = float(i*i + j + 0.1f*k); break;
                    default:v[p] = float((i+1)*(j+1) - (int)k); break;
                }
            }
        }
    }
}

// Launch helper
static void launch_gpu(const float* d_in, float* d_out, unsigned N,
                       float alpha, float dt, float dx,
                       dim3 grid, dim3 block)
{
    heat_step_kernel<<<grid, block>>>(d_in, d_out, N, alpha, dt, dx);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");
}

struct Case {
    unsigned N;
    int steps;
    const char* name;
};

int main() {
    printf("heat-3d-single-turn tests\n");

    // Stable CFL (dt <= dx^2 / (6*alpha)). We pick 0.8 of that.
    const float alpha = 0.01f;
    const float dx    = 1.0f;
    const float dt    = 0.8f * (dx*dx) / (6.0f * alpha);

    // Block configurations to exercise indexing behaviors
    const dim3 blocks[] = {
        dim3(8,8,8),       // 512 threads
        dim3(16,4,4),      // 256 threads
        dim3(32,2,2),      // 128 threads
    };

    const Case cases[] = {
        {1,  1, "N=1"},     // no interior
        {2,  1, "N=2"},     // no interior
        {3,  1, "N=3"},
        {8,  1, "N=8"},
        {17, 1, "N=17 (prime-ish)"},
        {32, 3, "N=32 (3 steps)"},
        {48, 3, "N=48 (3 steps)"},
    };

    int total = 0, passed = 0;

    for (const auto& c : cases) {
        for (int pat=0; pat<4; ++pat) {
            for (const auto& bdim : blocks) {

                const unsigned N = c.N;
                const size_t   NN = size_t(N)*N*N;
                const int      steps = c.steps;

                // Host buffers
                std::vector<float> h_in(NN), h_ref(NN), h_out(NN);
                fill_pattern(h_in, N, pat);

                // CPU oracle multi-step
                std::vector<float> cur = h_in, nxt(NN);
                for (int s=0;s<steps;s++) {
                    heat_step_cpu(cur, nxt, N, alpha, dt, dx);
                    cur.swap(nxt);
                }
                h_ref = cur;

                // Canary padding for device buffers (detect OOB)
                const size_t GUARD = 4096; // elements
                const float  SENT  = 1337.0f;

                float *d_buf_in = nullptr, *d_buf_out = nullptr;
                checkCuda(cudaMalloc(&d_buf_in,  (NN + 2*GUARD) * sizeof(float)), "malloc in");
                checkCuda(cudaMalloc(&d_buf_out, (NN + 2*GUARD) * sizeof(float)), "malloc out");

                // Fill guards with SENT
                checkCuda(cudaMemset(d_buf_in,  0, (NN+2*GUARD)*sizeof(float)), "memset in tmp");
                checkCuda(cudaMemset(d_buf_out, 0, (NN+2*GUARD)*sizeof(float)), "memset out tmp");

                // Host temp with guards for exact values
                std::vector<float> h_in_guard(NN + 2*GUARD, SENT);
                std::copy(h_in.begin(), h_in.end(), h_in_guard.begin()+GUARD);

                std::vector<float> h_out_guard(NN + 2*GUARD, SENT);

                // Upload guarded input and output
                checkCuda(cudaMemcpy(d_buf_in, h_in_guard.data(),
                                     (NN+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),
                                     "upload in guard");
                checkCuda(cudaMemcpy(d_buf_out, h_out_guard.data(),
                                     (NN+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),
                                     "upload out guard");

                // Alias interior pointers
                float* d_in  = d_buf_in  + GUARD;
                float* d_out = d_buf_out + GUARD;

                // Grid dims
                auto cdiv = [](unsigned a, unsigned b){ return (a + b - 1) / b; };
                dim3 grid(cdiv(N, bdim.x), cdiv(N, bdim.y), cdiv(N, bdim.z));

                // Multi-step on GPU (ping-pong)
                // d_in starts with h_in; after each step swap roles
                for (int s=0;s<steps;s++) {
                    launch_gpu(d_in, d_out, N, alpha, dt, dx, grid, bdim);
                    std::swap(d_in, d_out);
                }

                // After the loop, d_in always points to the latest results due to the swap above.
                float* d_result = d_in;

                // Download both guarded buffers so we can (a) extract results and (b) check canaries
                checkCuda(cudaMemcpy(h_out_guard.data(), d_buf_out,
                                     (NN+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),
                                     "download out guard");
                checkCuda(cudaMemcpy(h_in_guard.data(), d_buf_in,
                                     (NN+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),
                                     "download in guard");

                // Choose the correct interior based on which raw buffer d_in aliases.
                if (d_result == (d_buf_out + GUARD)) {
                    std::copy(h_out_guard.begin()+GUARD, h_out_guard.begin()+GUARD+NN, h_out.begin());
                } else {
                    std::copy(h_in_guard.begin()+GUARD, h_in_guard.begin()+GUARD+NN, h_out.begin());
                }

                // Validate:
                bool ok = true; ++total;

                // Helper: step-aware epsilon
                auto eps_for = [&](int steps) {
                    // Base 1e-4 for single step. For multi-step, allow modest inflation.
                    return (steps <= 1) ? 1e-4f : 5e-3f;
                };

                // Helper: max error diagnostics on failure
                auto dump_max_err = [&](const char* tag){
                    size_t worst = 0; float amax=0.f, rmax=0.f;
                    for (size_t i=0;i<NN;++i){
                        float a = std::fabs(h_out[i]-h_ref[i]);
                        float d = std::max(std::fabs(h_ref[i]), 1.0f);
                        float r = a/d;
                        if (a>amax){ amax=a; worst=i; }
                        if (r>rmax){ rmax=r; }
                    }
                    fprintf(stderr,"%s: maxAbs=%.6g maxRel=%.6g at idx=%zu\n", tag, amax, rmax, worst);
                };

                // Helper: face invariance check (copy-through boundary must match exactly)
                auto faces_equal = [&](float tol)->bool{
                    auto idx=[&](unsigned i,unsigned j,unsigned k){return I(i,j,k,N);};
                    for(unsigned i=0;i<N;i++)for(unsigned j=0;j<N;j++){
                        if (std::fabs(h_out[idx(i,j,0)]   - h_in[idx(i,j,0)])   > tol) return false;
                        if (std::fabs(h_out[idx(i,j,N-1)] - h_in[idx(i,j,N-1)]) > tol) return false;
                    }
                    for(unsigned i=0;i<N;i++)for(unsigned k=0;k<N;k++){
                        if (std::fabs(h_out[idx(i,0,k)]   - h_in[idx(i,0,k)])   > tol) return false;
                        if (std::fabs(h_out[idx(i,N-1,k)] - h_in[idx(i,N-1,k)]) > tol) return false;
                    }
                    for(unsigned j=0;j<N;j++)for(unsigned k=0;k<N;k++){
                        if (std::fabs(h_out[idx(0,j,k)]   - h_in[idx(0,j,k)])   > tol) return false;
                        if (std::fabs(h_out[idx(N-1,j,k)] - h_in[idx(N-1,j,k)]) > tol) return false;
                    }
                    return true;
                };

                // 1) CPU oracle with step-aware tolerance
                const float eps = eps_for(steps);
                bool oracle_ok = almost_equal(h_out, h_ref, eps);
                if (!oracle_ok) dump_max_err("ORACLE_DIFF");
                ok = ok && oracle_ok;

                // 1b) Face invariance check (boundaries must copy through exactly)
                ok = ok && faces_equal(0.0f);

                // 2) Input immutability: only check for single-step cases
                // Multi-step ping-pong inherently reuses buffers as input/output alternately
                bool immutability_ok = true;
                if (steps == 1) {
                    // interior
                    std::vector<float> h_in_orig(NN);
                    std::copy(h_in_guard.begin()+GUARD, h_in_guard.begin()+GUARD+NN, h_in_orig.begin());
                    immutability_ok = almost_equal(h_in_orig, std::vector<float>(h_in.begin(), h_in.end()), 1e-6f);

                    // guards unchanged
                    auto guard_ok = [&](const std::vector<float>& g){
                        for (size_t t=0;t<GUARD;t++){
                            if (g[t] != SENT) return false;
                            if (g[g.size()-1-t] != SENT) return false;
                        }
                        return true;
                    };
                    immutability_ok = immutability_ok && guard_ok(h_in_guard);
                }
                ok = ok && immutability_ok;

                // 3) Out-of-bounds writes: out guards must remain SENT
                {
                    auto guard_ok = [&](const std::vector<float>& g){
                        for (size_t t=0;t<GUARD;t++){
                            if (g[t] != SENT) return false;
                            if (g[g.size()-1-t] != SENT) return false;
                        }
                        return true;
                    };
                    ok = ok && guard_ok(h_out_guard);
                }

                printf("Case %-18s | pat=%d | block=(%2u,%2u,%2u) -> %s\n",
                       c.name, pat, bdim.x, bdim.y, bdim.z, ok ? "OK" : "FAIL");

                if (ok) ++passed;

                cudaFree(d_buf_in);
                cudaFree(d_buf_out);
            }
        }
    }

    printf("Summary: %d / %d tests passed\n", passed, total);
    if (passed == total) {
        printf("All tests passed.\n");
    } else {
        printf("Some tests FAILED.\n");
    }
    return (passed == total) ? 0 : 1;
}