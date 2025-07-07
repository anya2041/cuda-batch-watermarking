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

        float blended = input[idx] + 0.4f * watermark[wm_idx];
        output[idx] = (unsigned char)fminf(255.0f, blended);

    // DEBUG mode: forcibly overwrite to white if strong watermark
    // if (watermark[wm_idx] > 128) {
    //     output[idx] = 255;
    // }
    }
}

bool loadPGM(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    std::getline(file, line); // Expecting "P2"
    if (line != "P2") return false;

    // Skip comments
    do {
        std::getline(file, line);
    } while (line[0] == '#');

    std::istringstream dim(line);
    dim >> width >> height;

    int maxval;
    file >> maxval;

    data.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        int pixel;
        file >> pixel;
        data[i] = static_cast<unsigned char>(pixel);
    }

    return true;
}

void savePGM(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename);
    file << "P2\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file << static_cast<int>(data[i]) << " ";
        if ((i + 1) % width == 0) file << "\n";
    }
}

int main() {
    int img_w, img_h, wm_w, wm_h;
    std::vector<unsigned char> input_host, watermark_host;

    // Load input image
    if (!loadPGM("input/venus2.ascii.pgm", input_host, img_w, img_h)) {
        std::cerr << "Failed to load input image\n";
        return 1;
    }
    std::cout << "Loaded input image: " << img_w << " x " << img_h << "\n";

    // Load watermark
    if (!loadPGM("input/watermark.pgm", watermark_host, wm_w, wm_h)) {
        std::cerr << "Failed to load watermark\n";
        return 1;
    }
    std::cout << "Loaded watermark image: " << wm_w << " x " << wm_h << "\n";

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


    savePGM("output/watermarked.pgm", output_host, img_w, img_h);
    std::cout << "Watermark applied! Check output/watermarked.pgm\n";

    cudaFree(d_input);
    cudaFree(d_watermark);
    cudaFree(d_output);

    return 0;
}
