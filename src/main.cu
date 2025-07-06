#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WIDTH 256
#define HEIGHT 256
#define IMAGE_SIZE (WIDTH * HEIGHT)

// CUDA Kernel: Embed watermark (simple brightness adjustment)
__global__ void watermarkKernel(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = min(255, image[idx] + 50);  // simple brightening watermark
    }
}

// Read PGM image (P5 format)
unsigned char* loadPGM(const std::string& filename, int* width, int* height) {
    std::ifstream ifs(filename, std::ios::binary);
    std::string header;
    int maxval;

    ifs >> header;
    if (header != "P5") {
        std::cerr << "Unsupported format. Use P5." << std::endl;
        exit(1);
    }

    ifs >> *width >> *height >> maxval;
    ifs.ignore(); // consume newline

    unsigned char* data = new unsigned char[(*width) * (*height)];
    ifs.read(reinterpret_cast<char*>(data), (*width) * (*height));
    ifs.close();
    return data;
}

// Save PGM image
void savePGM(const std::string& filename, const unsigned char* data, int width, int height) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P5\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data), width * height);
    ofs.close();
}

int main() {
    int width, height;
    unsigned char* h_image = loadPGM("watermark.pgm", &width, &height);

    unsigned char* d_image;
    cudaMalloc(&d_image, width * height);
    cudaMemcpy(d_image, h_image, width * height, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    watermarkKernel<<<gridDim, blockDim>>>(d_image, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_image, d_image, width * height, cudaMemcpyDeviceToHost);
    savePGM("output/watermarked.pgm", h_image, width, height);

    std::cout << "\x1B[32m\x1B[1mWatermarked sample pixel [0]: " << (int)h_image[0] << "\x1B[0m" << std::endl;

    cudaFree(d_image);
    delete[] h_image;
    return 0;
}
