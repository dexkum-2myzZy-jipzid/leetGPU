#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N/2){
        float tmp = input[i];
        input[i] = input[N-1-i];
        input[N-1-i] = tmp;
    }

}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}