#include "solve.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 A 的一块
        if (row < M && t*TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t*TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // 加载 B 的一块
        if (col < K && t*TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t*TILE_SIZE + ty)*K + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // 累加内积
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // 写回 C
    if (row < M && col < K)
        C[row * K + col] = acc;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
