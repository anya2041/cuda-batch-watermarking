// src/main.cu
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    }

__global__ void apply_watermark(unsigned char* input, unsigned char* watermark, unsigned char* output,
                                int width, int height, int wm_width, int wm_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx]; // default: copy original

    if (x < wm_width && y < wm_height) {
        int wm_idx = y * wm_width + x;
        output[idx] = 0.5f * input[idx] + 0.5f * watermark[wm_idx];
    }
}

bool loadPGM(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    std::getline(file, line); // P2 or P5
    if (line != "P2") return false;

    do { std::getline(file, line); } while (line[0] == '#'); // skip comments
    std::istringstream dim(line);
    dim >> width >> height;

    int maxval;
    file >> maxval;

    data.resize(width * height);
    for (int i = 0; i < width * height; ++i) file >> (int&)data[i];
    return true;
}

void savePGM(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename);
    file << "P2\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file << (int)data[i] << " ";
        if ((i + 1) % width == 0) file << "\n";
    }
}

int main() {
    int img_w, img_h, wm_w, wm_h;
    std::vector<unsigned char> input_host, watermark_host;

    if (!loadPGM("input/mona_lisa.ascii.pgm", input_host, img_w, img_h)) {
        std::cerr << "Failed to load input image\n";
        return 1;
    }
    if (!loadPGM("watermark.pgm", watermark_host, wm_w, wm_h)) {
        std::cerr << "Failed to load watermark\n";
        return 1;
    }

    size_t image_size = img_w * img_h * sizeof(unsigned char);
    size_t watermark_size = wm_w * wm_h * sizeof(unsigned char);

    unsigned char *d_input, *d_watermark, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, image_size));
    CHECK_CUDA(cudaMalloc(&d_watermark, watermark_size));
    CHECK_CUDA(cudaMalloc(&d_output, image_size));

    CHECK_CUDA(cudaMemcpy(d_input, input_host.data(), image_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_watermark, watermark_host.data(), watermark_size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((img_w + 15) / 16, (img_h + 15) / 16);

    apply_watermark<<<blocks, threads>>>(d_input, d_watermark, d_output, img_w, img_h, wm_w, wm_h);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<unsigned char> output_host(img_w * img_h);
    CHECK_CUDA(cudaMemcpy(output_host.data(), d_output, image_size, cudaMemcpyDeviceToHost));

    savePGM("output/mona_lisa_watermarked.pgm", output_host, img_w, img_h);

    cudaFree(d_input);
    cudaFree(d_watermark);
    cudaFree(d_output);

    std::cout << "Watermark applied! Check output/mona_lisa_watermarked.pgm" << std::endl;
    return 0;
}