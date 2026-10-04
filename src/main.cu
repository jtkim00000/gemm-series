/*
* GEMM Kernel Optimization Project
* Experimental Throughput Analysis
* Projeject by Jesse Kim
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <cublas_v2.h>

#include "benchmark/benchmark.cuh"
#include "matmul/naive_matmul.cuh"
#include "matmul/tiled_matmul.cuh"
#include "matmul/thread_coarsened_matmul.cuh"
#include "matmul/coalesced_matmul.cuh"
#include "matmul/peak_matmul.cuh"

struct MatrixShape {
    int M;
    int N;
    int K;
};

int main() {

    cudaError_t err = cudaSetDevice(0);

    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "Using GPU: " << prop.name << '\n';
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << '\n';

    cublasHandle_t cublas_handle;

    cublasStatus_t status = cublasCreate(&cublas_handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed: "
                  << status << '\n';
        return 1;
    }

    std::cout << "cuBLAS initialized successfully\n";

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    std::cout << "==================================================" << '\n';
    std::cout << "     General Matrix Multiplication Kernels        " << '\n';
    std::cout << "     Experimental throughput analysis             " << '\n';
    std::cout << "      - RTX4060 GPU CUDA Kernels                  " << '\n';
    std::cout << "      - C++17                                     " << '\n';
    std::cout << "     Project by: Jesse Kim                        " << '\n';
    std::cout << "==================================================" << '\n';

    std::cout << std::left 
              << std::setw(18) << "Shape (M x N x K)" 
              << std::setw(16) << "Naive (TFLOPS)" 
              << std::setw(16) << "Tiled (TFLOPS)" 
              << std::setw(16) << "TC (TFLOPS)" 
              << std::setw(16) << "Coalesced (TFLOPS)" 
              << std::setw(16) << "Peak (TFLOPS)" 
              << std::setw(16) << "cuBLAS (TFLOPS)" << "\n";
    std::cout << std::string(98, '-') << "\n";

    std::vector<MatrixShape> test_shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 512, 1024},  // Non-square shape
        {512, 4096, 2048}   // Non-square shape
    };

    constexpr int BLOCK_SIZE{16};
    constexpr int COARSE_FACTOR{4};
    constexpr int ITERS_PER_SHAPE{100};

    for (const auto& shape : test_shapes) {
        int M{shape.M};
        int N{shape.N};
        int K{shape.K};

        size_t size_A{static_cast<size_t>(M) * K * sizeof(float)};
        float* d_A{nullptr};
        cudaMalloc(reinterpret_cast<void**>(&d_A), size_A);

        // Matrix B: K x N
        size_t size_B{static_cast<size_t>(K) * N * sizeof(float)};
        float* d_B{nullptr};
        cudaMalloc(reinterpret_cast<void**>(&d_B), size_B);

        // Matrix C: M x N
        size_t size_C{static_cast<size_t>(M) * N * sizeof(float)};
        float* d_C{nullptr};
        cudaMalloc(reinterpret_cast<void**>(&d_C), size_C);

        cudaMemset(d_A, 0, size_A);
        cudaMemset(d_B, 0, size_B);
        cudaMemset(d_C, 0, size_C);

        double total_flops{2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)};

        double t_naive = benchmarkKernel([&]() {
            naiveMatmulGPU<BLOCK_SIZE>(d_A, d_B, d_C, M, N, K);
        }, ITERS_PER_SHAPE);

        double t_tiled = benchmarkKernel([&]() {
            tiledMatmulGPU<BLOCK_SIZE>(d_A, d_B, d_C, M, N, K);
        }, ITERS_PER_SHAPE);

        double t_tc = benchmarkKernel([&]() {
            threadCoarsenedMatmulGPU<BLOCK_SIZE, COARSE_FACTOR>(d_A, d_B, d_C, M, N, K);
        }, ITERS_PER_SHAPE);

        double t_coalesced = benchmarkKernel([&]() {
            coalescedMatmulGPU<BLOCK_SIZE, COARSE_FACTOR>(d_A, d_B, d_C, M, N, K);
        }, ITERS_PER_SHAPE);

        double t_peak = benchmarkKernel([&]() {
            peakMatmulGPU<BLOCK_SIZE, COARSE_FACTOR>(d_A, d_B, d_C, M, N, K);
        }, ITERS_PER_SHAPE);

        double t_cublas = benchmarkKernel([&]() {
            // Compute Row-Major A x B by calling cuBLAS on Column-Major B x A
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        N, M, K, 
                        &alpha, d_B, N, 
                        d_A, K, 
                        &beta, d_C, N);
        }, ITERS_PER_SHAPE);

        // Calculate GFLOPS/s for current shape
        double tflops_naive{(total_flops / t_naive) / 1e12};
        double tflops_tiled{(total_flops / t_tiled) / 1e12};
        double tflops_tc{(total_flops / t_tc) / 1e12};
        double tflops_coalesced{(total_flops / t_coalesced) / 1e12};
        double tflops_peak{(total_flops / t_peak) / 1e12};
        double tflops_cublas{(total_flops / t_cublas) / 1e12};

        std::string shape_str = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        std::cout << std::fixed << std::setprecision(3)
                << std::setw(18) << shape_str
                << std::setw(16) << tflops_naive
                << std::setw(16) << tflops_tiled
                << std::setw(16) << tflops_tc
                << std::setw(16) << tflops_coalesced
                << std::setw(16) << tflops_peak
                << std::setw(16) << tflops_cublas << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cublasDestroy(cublas_handle);

    return 0;
}