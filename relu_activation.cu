#include "solve.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N){
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    dim3 blockDim1(256);
    dim3 gridDim1((N+blockDim1.x-1)/blockDim1.x);

    relu_kernel<<<gridDim1, blockDim1>>>(input, output, N);
    cudaDeviceSynchronize();
}
