#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>

const int QUERIES_PER_BLOCK = 32;
const int KEYS_PER_TILE = 32;
const int MAX_DK  = 128;

__global__ void mhsa_kernel(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h){

    const int head_id = blockIdx.x;
    const int query_base = blockIdx.y * QUERIES_PER_BLOCK;
    const int lane_id = threadIdx.x; //[0,31] one wrap
    const int query_row = query_base + lane_id;

    const int d_k = d_model / h; // single head dim

    if (d_k > MAX_DK) return;

    const float scale = rsqrtf(static_cast<float>(d_k)); // 1/√d_k

    extern __shared__ float shmem[]; // prev 32 cols: K, later 32 cols: V
    float* K_tile = shmem;
    float* V_tile = shmem + KEYS_PER_TILE * d_k;
    
    float q_vec[MAX_DK]; // current query vector  (d_k floats)
    if(query_row < N){
        const float* q_ptr = Q + query_row * d_model + head_id * d_k; //[row, head, 0]
        #pragma unroll
        for(int i = 0; i < d_k; i++)
            q_vec[i] = q_ptr[i];
    }else{
        #pragma unroll
        for(int i =0; i < d_k; ++i) q_vec[i] = 0.f;
    }

    // register 
    // Streaming softmax init
    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    float weighted_v[MAX_DK] = {0.f};

    for(int key_base = 0; key_base < N; key_base += KEYS_PER_TILE){
        const int keys_this_tile = min(KEYS_PER_TILE, N - key_base);

        const int elems = keys_this_tile * d_k;
        for (int idx = lane_id; idx < elems; idx += 32) {
            int k   = idx / d_k;                            
            int dim = idx % d_k;                           
            int gk  = key_base + k;                         
        
            K_tile[idx] = K[gk * d_model + head_id * d_k + dim]; 
            V_tile[idx] = V[gk * d_model + head_id * d_k + dim];
        }

        __syncthreads();

        if (query_row < N) {
            for (int k = 0; k < keys_this_tile; ++k){
                /* 1.dot(q , k) */
                const float* k_vec = &K_tile[k * d_k];
                const float* v_vec = &V_tile[k * d_k];

                float score = 0.f;
                #pragma unroll
                for (int i = 0; i < d_k; ++i)
                    score += q_vec[i] * k_vec[i];
                score *= scale; //score = (q · k) / √d_k

                /* 2.streaming-softmax  */
                if (score > running_max) {
                    float old_scale = expf(running_max - score);   // e^{m_old-m_new}
                    running_sum = running_sum * old_scale + 1.f;
                    #pragma unroll
                    for (int i = 0; i < d_k; ++i)
                        weighted_v[i] = weighted_v[i] * old_scale + v_vec[i];
                    running_max = score;
                } else {                                           // 仍用旧 max
                    float w = expf(score - running_max);           // e^{x-m}
                    running_sum += w;
                    #pragma unroll
                    for (int i = 0; i < d_k; ++i)
                        weighted_v[i] += w * v_vec[i];
                }
            }
            
        }
        __syncthreads();
    }

    if (query_row < N) {
        float norm = 1.f / running_sum;
        float* out_ptr = output + query_row * d_model + head_id * d_k;
    
        #pragma unroll
        for (int i = 0; i < d_k; ++i) 
            out_ptr[i] = weighted_v[i] * norm;
    }

}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    const int d_k = d_model / h;
    dim3 block_dim(QUERIES_PER_BLOCK);
    dim3 grid_dim(h, ( N + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK);

    size_t shared_bytes = KEYS_PER_TILE * d_k * 2 * sizeof(float); // k_tile + v_tile

    mhsa_kernel<<<grid_dim, block_dim, shared_bytes>>>(Q, K, V, output, N, d_model, h);
}
