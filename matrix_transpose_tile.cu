#include "solve.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
    int y = blockIdx.y * blockDim.y + threadIdx.y; // rows
    
    if(x < cols && y < rows){
        // input[row][col] = input[y * cols + x]
        tile[threadIdx.x][threadIdx.y] = input[y * cols + x];
    }else{
        tile[threadIdx.x][threadIdx.y] = 0.0f;
    }

    __syncthreads();

    if(x < cols && y < rows){
        // output[col][row] = output[col * rows + row]
        output[x * rows + y] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
                       (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}