#include "solve.h"
#include <cuda_runtime.h>

__global__ void block_reduce_kernel(const float *input, float *d_block_sums, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local = 0.0f;
    for(int i = pos; i < N; i+=stride){
        local += input[i];
    }

    sdata[tid] = local;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0) d_block_sums[blockIdx.x] = sdata[0];
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shmem = threads * sizeof(float);

    // alloc memory
    float *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // init output 0.0
    cudaMemset(output, 0.0f, sizeof(float));

    // call kernel func
    block_reduce_kernel<<<blocks, threads, shmem>>>(input, d_block_sums, N);

    block_reduce_kernel<<<1, threads, shmem>>>(d_block_sums, output, blocks);

    // clearn memory
    cudaFree(d_block_sums);
    cudaDeviceSynchronize();
}