// mhsa_readable.cu — tile‑based Multi‑Head Self‑Attention (A100)
// -----------------------------------------------------------------------------
//  ▸ Purpose
//    Educational reference showing how to implement Flash‑style streaming
//    soft‑max on NVIDIA A100.  The code is intentionally *one kernel* and keeps
//    arithmetic simple so that each step is easy to follow.
//
//  ▸ Assumptions
//      Q, K, V, O : row‑major [N, d_model] on device
//      d_k = d_model / h ≤ 128   (fits in registers)
//      ONE WARP (32 threads)  ↔  ONE BLOCK
//          • each thread processes one query row
//          • queries per block (TQ) = 32  (blockDim.x)
//      TILE_K = 32 keys loaded per iteration into shared memory
//
//  ▸ Notation
//      N      – sequence length  (rows)
//      d_model– embedding size   (columns)
//      h      – number of heads
//      d_k    – head dimension   (columns per head)   = d_model / h
//
//  ▸ Algorithm (per head)
//      for each query‑tile (32 rows)
//          for each key‑tile (32 rows)
//              – cooperatively load K & V tile to shared memory
//              – each thread streams over keys, updates running
//                  {max, sumExp, weightedV}
//      write   output = weightedV / sumExp
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

// -----------------------------------------------------------------------------
// Tunable tile sizes (powers of two help alignment & occupancy)
// -----------------------------------------------------------------------------
constexpr int QUERIES_PER_BLOCK = 32;   // 1 warp ↔ 1 query tile
constexpr int KEYS_PER_TILE     = 32;   // keys loaded per iteration
constexpr int MAX_DK            = 128;  // maximum head dimension supported

// -----------------------------------------------------------------------------
// Kernel: one block  ⇨  (single head, 32 query rows)
// -----------------------------------------------------------------------------
__global__ void mhsa_kernel(const float* __restrict__ Q,
                            const float* __restrict__ K,
                            const float* __restrict__ V,
                            float*       __restrict__ O,
                            int N, int d_model, int num_heads)
{
    /* -------------------------- index bookkeeping -------------------------- */
    const int head_id      = blockIdx.x;                       // 0 … h‑1
    const int query_base   = blockIdx.y * QUERIES_PER_BLOCK;   // first query row this block handles
    const int lane_id      = threadIdx.x;                      // 0 … 31   (one lane ↔ one query)
    const int query_row    = query_base + lane_id;             // global query index

    const int d_k = d_model / num_heads;                       // head dimension
    if (d_k > MAX_DK) return;                                  // guard (demo scope)

    /* ------------------------- per‑thread registers ------------------------ */
    float q_vec[MAX_DK];               //    current query vector  (d_k floats)
    float weighted_v[MAX_DK] = {0.f};  // Σ softmax * V           (d_k floats)

    float running_max  = -FLT_MAX; // streaming soft‑max state: max(scores)
    float running_sum  = 0.f;           //                     Σ exp(score – max)
    const float scale  = rsqrtf(static_cast<float>(d_k)); // 1/√d_k  (dot‑prod scaling)

    /* ------------------------- load query into regs ------------------------ */
    if (query_row < N) {
        const float* q_ptr = Q + query_row * d_model + head_id * d_k;
        #pragma unroll
        for (int i = 0; i < d_k; ++i) q_vec[i] = q_ptr[i];
    } else {
        // out‑of‑range dummy row – fill with zeros to keep arithmetic valid
        #pragma unroll
        for (int i = 0; i < d_k; ++i) q_vec[i] = 0.f;
    }

    /* ---------------------- shared memory for K / V tile ------------------- */
    extern __shared__ float shmem[];                       // dynamic allocation
    float* __restrict__ K_tile = shmem;                    // size = KEYS_PER_TILE * d_k
    float* __restrict__ V_tile = K_tile + KEYS_PER_TILE * d_k;

    /* -------------------------- iterate over key tiles --------------------- */
    for (int key_base = 0; key_base < N; key_base += KEYS_PER_TILE) {
        const int keys_this_tile = min(KEYS_PER_TILE, N - key_base);

        /* ---- cooperative load K & V rows of this tile into shared mem ---- */
        const int elements_in_tile = keys_this_tile * d_k;
        for (int idx = lane_id; idx < elements_in_tile; idx += QUERIES_PER_BLOCK) {
            int key_offset  = idx / d_k;   // which key in tile (0 … keys_this_tile‑1)
            int dim         = idx % d_k;   // which column therein
            int global_key  = key_base + key_offset;

            K_tile[idx] = K[global_key * d_model + head_id * d_k + dim];
            V_tile[idx] = V[global_key * d_model + head_id * d_k + dim];
        }
        __syncthreads();   // ensure tile is fully loaded before use

        /* ---- stream through keys in tile, update soft‑max stats ---------- */
        if (query_row < N) {
            for (int k = 0; k < keys_this_tile; ++k) {
                const float* __restrict__ k_vec = &K_tile[k * d_k];
                const float* __restrict__ v_vec = &V_tile[k * d_k];

                // dot‑product (q · k)
                float score = 0.f;
                #pragma unroll
                for (int i = 0; i < d_k; ++i) score += q_vec[i] * k_vec[i];
                score *= scale;   // scaled dot‑product

                if (score > running_max) {
                    // new maximum – rescale running sums (numerical stability)
                    float old_scale = expf(running_max - score);
                    running_sum = running_sum * old_scale + 1.f;
                    #pragma unroll
                    for (int i = 0; i < d_k; ++i)
                        weighted_v[i] = weighted_v[i] * old_scale + v_vec[i];
                    running_max = score;
                } else {
                    float w = expf(score - running_max);     // exp(score – max)
                    running_sum += w;
                    #pragma unroll
                    for (int i = 0; i < d_k; ++i)
                        weighted_v[i] += w * v_vec[i];
                }
            }
        }
        __syncthreads();   // release shared mem before loading next tile
    }

    /* ------------------------ write normalized output ---------------------- */
    if (query_row < N) {
        float norm = 1.f / running_sum;
        float* out_ptr = O + query_row * d_model + head_id * d_k;
        #pragma unroll
        for (int i = 0; i < d_k; ++i) out_ptr[i] = weighted_v[i] * norm;
    }
}

// -----------------------------------------------------------------------------
// Host entry: dispatch kernel with one warp per block
// -----------------------------------------------------------------------------
void solve(const float* Q, const float* K, const float* V,
                       float*      O, int N, int d_model, int h)
{
    const int d_k = d_model / h;

    dim3 grid_dim(h, (N + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK);
    dim3 block_dim(QUERIES_PER_BLOCK);   // 32 lanes = 1 warp

    size_t shared_bytes = KEYS_PER_TILE * d_k * 2 * sizeof(float);

    mhsa_kernel<<<grid_dim, block_dim, shared_bytes>>>(Q, K, V, O, N, d_model, h);
}
