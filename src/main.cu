#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)      \
                      << " at " << __FILE__ << ':' << __LINE__        \
                      << std::endl;                                   \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

namespace fs = std::filesystem;

bool loadPGM_P2(const std::string &filename,
                std::vector<unsigned char> &data,
                int &w, int &h) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line); // "P2"
    if (line != "P2") return false;

    do {
        std::getline(file, line);
    } while (line[0] == '#');

    std::istringstream dim(line);
    dim >> w >> h;

    int maxval;
    file >> maxval;

    data.resize(w * h);
    for (int i = 0; i < w * h; ++i) {
        int pixel;
        file >> pixel;
        data[i] = static_cast<unsigned char>(pixel);
    }
    return true;
}

bool savePGM_P2(const std::string &filename,
                const std::vector<unsigned char> &data,
                int w, int h) {
    std::ofstream f(filename);
    if (!f) return false;
    f << "P2\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        f << static_cast<int>(data[i]) << " ";
        if ((i + 1) % w == 0) f << "\n";
    }
    return true;
}

__global__ void apply_watermark(const unsigned char* input,
                                unsigned char* output,
                                const unsigned char* watermark,
                                int img_w, int img_h,
                                int wm_w, int wm_h,
                                float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h) return;

    int idx = y * img_w + x;
    output[idx] = input[idx];

    if (x < wm_w && y < wm_h) {
        int wm_idx = y * wm_w + x;
        float blend = (1.0f - alpha) * input[idx] + alpha * watermark[wm_idx];
        output[idx] = static_cast<unsigned char>(fminf(255.0f, blend));
    }
}

int main() {
    std::string input_dir = "input";
    std::string output_dir = "output";
    std::string watermark_file = "input/watermark.pgm";

    int wm_w, wm_h;
    std::vector<unsigned char> wm_host;
    if (!loadPGM_P2(watermark_file, wm_host, wm_w, wm_h)) {
        std::cerr << "Failed to load watermark.pgm\n";
        return 1;
    }

    size_t wm_size = wm_w * wm_h * sizeof(unsigned char);
    unsigned char* d_wm;
    CHECK_CUDA(cudaMalloc(&d_wm, wm_size));
    CHECK_CUDA(cudaMemcpy(d_wm, wm_host.data(), wm_size, cudaMemcpyHostToDevice));

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".pgm") continue;

        std::string in_path = entry.path().string();
        std::string out_path = output_dir + "/" + entry.path().stem().string() + "_wm.pgm";

        int img_w, img_h;
        std::vector<unsigned char> img_host;
        if (!loadPGM_P2(in_path, img_host, img_w, img_h)) {
            std::cerr << "Failed to load image: " << in_path << "\n";
            continue;
        }

        size_t img_size = img_w * img_h * sizeof(unsigned char);
        unsigned char *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, img_size));
        CHECK_CUDA(cudaMalloc(&d_out, img_size));

        CHECK_CUDA(cudaMemcpy(d_in, img_host.data(), img_size, cudaMemcpyHostToDevice));

        dim3 threads(16, 16);
        dim3 blocks((img_w + 15) / 16, (img_h + 15) / 16);

        apply_watermark<<<blocks, threads>>>(
            d_in, d_out, d_wm,
            img_w, img_h, wm_w, wm_h, 0.4f);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(img_host.data(), d_out, img_size, cudaMemcpyDeviceToHost));
        savePGM_P2(out_path, img_host, img_w, img_h);

        cudaFree(d_in);
        cudaFree(d_out);
    }

    cudaFree(d_wm);
    std::cout << "Done watermarking all images.\n";
    return 0;
}
