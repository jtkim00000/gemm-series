#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <chrono>
#include <cuda_runtime.h>

// Generic benchmark runner template
template <typename KernelFunc>
double benchmarkKernel(KernelFunc&& kernel, int iterations) {
    // Warm-up run
    kernel();
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    return diff.count() / iterations; // Average time per run in seconds
}

#endif