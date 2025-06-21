#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>

__global__ void reduce_max_kernel(const float * input, float* output, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local = -FLT_MAX;
    for(int i = pos; i < N; i += stride){
        local = fmaxf(local, input[i]);
    }

    sdata[tid] = local;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void exp_kernel(const float* input, float* output, const float* max_val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = expf(input[i] - *max_val);
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local = 0.0f;
    for(int i = pos; i < N; i += stride){
        local += input[i]; 
    }

    sdata[tid] = local;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void softmax_kernel(const float* exp_values, float* output, const float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = exp_values[i] / *sum; 
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shmem = threads * sizeof(float);

    // get max val
    float *d_block_maxs, *max;
    cudaMalloc(&d_block_maxs, blocks * sizeof(float));
    cudaMalloc(&max, sizeof(float));
    
    float init_max = -FLT_MAX;
    cudaMemcpy(max, &init_max, sizeof(float), cudaMemcpyHostToDevice);

    reduce_max_kernel<<<blocks, threads, shmem>>>(input, d_block_maxs, N);
    reduce_max_kernel<<<1, threads, shmem>>>(d_block_maxs, max, blocks);

    // get every elements exp(val - max)
    float *d_exp_values;
    cudaMalloc(&d_exp_values, N * sizeof(float));
    
    exp_kernel<<<blocks, threads>>>(input, d_exp_values, max, N);

    // sum exp values
    float *sum;
    cudaMalloc(&sum, sizeof(float));
    
    float init_sum = 0.0f;
    cudaMemcpy(sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_kernel<<<blocks, threads, shmem>>>(d_exp_values, d_block_maxs, N);
    reduce_sum_kernel<<<1, threads, shmem>>>(d_block_maxs, sum, blocks);

    // softmax
    softmax_kernel<<<blocks, threads>>>(d_exp_values, output, sum, N);

    cudaDeviceSynchronize();
    
    cudaFree(d_block_maxs);
    cudaFree(d_exp_values);
    cudaFree(max);
    cudaFree(sum);
}