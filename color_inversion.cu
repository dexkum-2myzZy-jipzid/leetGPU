#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x < width && pixel_y < height) {
        int base = (pixel_y * width + pixel_x) * 4;
        image[base + 0] = 255 - image[base + 0];  // R
        image[base + 1] = 255 - image[base + 1];  // G
        image[base + 2] = 255 - image[base + 2];  // B
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    dim3 threadsPerBlock(16, 16); // 2 Dim
    dim3 blocksPerGrid((width + 15) / 16,   // handle width pixel
                       (height + 15) / 16); // handle height pixel
    
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}