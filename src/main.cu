// File: src/main.cu
#include <iostream>
#include <cuda_runtime.h>
#include "../include/watermark.h"

#define IMG_WIDTH 512
#define IMG_HEIGHT 512

__global__ void apply_watermark(unsigned char *image, unsigned char *watermark, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = (image[idx] >> 1) + (watermark[idx] >> 1); // blend
    }
}

int main() {
    int img_size = IMG_WIDTH * IMG_HEIGHT;

    unsigned char *h_image = new unsigned char[img_size];
    unsigned char *h_watermark = new unsigned char[img_size];

    for (int i = 0; i < img_size; ++i) {
        h_image[i] = 100;       // simulate input image
        h_watermark[i] = 50;    // simulate watermark
    }

    unsigned char *d_image, *d_watermark;
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_watermark, img_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_watermark, h_watermark, img_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((IMG_WIDTH + 15) / 16, (IMG_HEIGHT + 15) / 16);

    apply_watermark<<<numBlocks, threadsPerBlock>>>(d_image, d_watermark, IMG_WIDTH, IMG_HEIGHT);

    cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost);

    std::cout << "Watermarked sample pixel [0]: " << (int)h_image[0] << std::endl;

    cudaFree(d_image);
    cudaFree(d_watermark);
    delete[] h_image;
    delete[] h_watermark;

    return 0;
}
