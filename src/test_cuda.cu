#include <iostream>
#include <cuda_runtime.h>

int main() {
    int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);

    std::cout << "cudaGetDeviceCount: "
              << cudaGetErrorString(err) << '\n';

    std::cout << "Device count: " << count << '\n';

    if (count > 0) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);

        std::cout << "GPU: " << prop.name << '\n';
        std::cout << "Compute capability: "
                  << prop.major << "." << prop.minor << '\n';
    }

    return 0;
}
