#include "solve.h"
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 512;   // 典型 sweet-spot
constexpr float ZERO = 0.0f;

/* ---------------- 内核实现 ---------------- */
__global__ void reduce_sum_kernel(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  int N)
{
    extern __shared__ float sdata[];      // THREADS_PER_BLOCK × 4 B

    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x * 2 + tid; // 2×unroll
    float mySum = 0.0f;

    /* -------- ① 读全局内存到寄存器 -------- */
    if (idx < N)               mySum += in[idx];
    if (idx + blockDim.x < N)  mySum += in[idx + blockDim.x];

    /* -------- ② 写入 shared memory -------- */
    sdata[tid] = mySum;
    __syncthreads();

    /* -------- ③ Block 内规约 -------- */
    // s = 256,128,64 … (>32) 部分需要显式 barrier
    for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 剩下 32 个线程用 warp-level volatile 技巧
    if (tid < 32) {
        volatile float* v = sdata;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid +  8];
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
    }

    /* -------- ④ Block 0 号线程跨 Block 原子加 -------- */
    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

/* ---------------- 题目指定入口 ---------------- */
// input, output are **device** pointers
void solve(const float* input, float* output, int N)
{
    /* 0️⃣ 把 output 清零（在 GPU 端），保证结果可复用 */
    cudaMemset(output, 0, sizeof(float));

    /* 1️⃣ 计算 grid / block 尺寸 */
    int blocks = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    size_t sharedBytes = THREADS_PER_BLOCK * sizeof(float);

    /* 2️⃣ Launch */
    reduce_sum_kernel<<<blocks, THREADS_PER_BLOCK, sharedBytes>>>(input, output, N);

    /* 3️⃣ 同步（LeetGPU 通常要求） */
    cudaDeviceSynchronize();
}