#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// matmul QKT 
// Q: M*d, K: N*d, QKT: M*N
__global__ void matmul_QK_T_kernel(const float* Q, const float* K, float* QK_T, int M, int N, int d){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M

    if(row < M && col < N){
        float local = 0.0f;
        for(int i = 0; i < d; i++){
            local += Q[row * d + i] * K[col * d + i];
        }
        QK_T[row * N + col] = local;
    }
}

// softmax
// QKT: M*N
__global__ void softmax_kernel(const float* inp, float* out, int M, int N, int d){
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= M) return;
    
    const float* row_inp = inp + row * N;
    float* row_out = out + row * N;
    
    // 1. get max
    float scale = 1.0f / sqrtf(static_cast<float>(d));
    float max_val = -FLT_MAX;
    
    for(int i = tid; i < N; i += blockDim.x){
        max_val = fmaxf(max_val, row_inp[i] * scale);
    }
    sdata[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    
    // 2. sum exp(val - max)
    float sum = 0.0f;
    for(int i = tid; i < N; i += blockDim.x){
        sum += expf(row_inp[i] * scale - max_val);
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];
    
    // 3. compute final 
    for(int i = tid; i < N; i += blockDim.x){
        row_out[i] = expf(row_inp[i] * scale - max_val) / sum;
    }
}


// matmul P(M,N) * V(N,d) -> output(M,d)
__global__ void matmul_SV_kernel(const float* S, const float* V, float* output, int M, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // d

    if (row < M && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += S[row * N + i] * V[i * d + col];
        }
        output[row * d + col] = sum;
    }
}


// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // Q: M*d, K: N*d, V: N*d
    
    float *d_QK_T, *d_S;
    size_t mn_size = M*N * sizeof(float);
    cudaMalloc(&d_QK_T, mn_size);
    cudaMalloc(&d_S, mn_size);

    // matmul QKT => M*N
    dim3 blockDim1(16, 16);
    dim3 gridDim1((N + 15) / 16, (M + 15) / 16);
    matmul_QK_T_kernel<<<gridDim1, blockDim1>>>(Q, K, d_QK_T, M, N, d);
    cudaDeviceSynchronize();

    // softmax M*N /sqrt(d)
    dim3 blockDim2(256);
    dim3 gridDim2(M);
    size_t sh_mem = blockDim2.x * sizeof(float);
    softmax_kernel<<<gridDim2, blockDim2, sh_mem>>>(d_QK_T, d_S, M, N, d);
    cudaDeviceSynchronize();

    // S*V
    dim3 blockDim3(16, 16);
    dim3 gridDim3((d + 15) / 16, (M + 15) / 16);
    matmul_SV_kernel<<<gridDim3, blockDim3>>>(d_S, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(d_QK_T);
    cudaFree(d_S);
}