#include "solve.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N){
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > 65535) {
        threadsPerBlock = 1024;
        blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
