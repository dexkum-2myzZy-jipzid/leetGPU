#include "solve.h"
#include <cuda_runtime.h>

__global__ void block_reduce_kernel(const float* y_samples, float *d_block_sums, int n_samples){
    // blockDim.x size
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local = 0.0f;
    for (int i = pos; i < n_samples; i += stride){
        local += y_samples[i];
    }

    sdata[tid] = local;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    
    if(tid == 0) d_block_sums[blockIdx.x] = sdata[0];
}

__global__ void final_reduce_kernel(const float* d_block_sums, int block_sum_len, float* result, float scale){
    // blockDim.x size 256 
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local = 0.0f;
    for (int i = tid; i < block_sum_len ; i+=stride){
        local += d_block_sums[i];
    }

    sdata[tid] = local;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0 ; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) *result = sdata[0] * scale;
}

// y_samples, result are device pointers
void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    const int threads = 256;
    int blocks = (n_samples + threads - 1) / threads; // if n_samples = 100,000,000. blocks = 390625
    size_t shmen = threads * sizeof(float);
    
    // d_block_sums
    float *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    cudaMemset(result, 0, sizeof(float));

    // shared memory
    block_reduce_kernel<<<blocks, threads, shmen>>>(y_samples, d_block_sums, n_samples);

    const int final_thread = 256;
    float scale = (b-a) / n_samples;
    final_reduce_kernel<<<1, final_thread, shmen>>>(d_block_sums, blocks, result, scale);

    cudaFree(d_block_sums);
}

