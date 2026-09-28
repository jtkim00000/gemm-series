/*
* GEMM Kernel Optimization Project
* Experimental Throughput Analysis
*/

#include <iostream>

#include "matmul/naive_matmul.cuh"
#include "matmul/tiled_matmul.cuh"
#include "matmul/thread_coarsened_matmul.cuh"
#include "matmul/mao_matmu.cuh"


int main() {

    std::cout << "==================================================" << '\n';
    std::cout << "     General Matrix Multiplication Kernels        " << '\n';
    std::cout << "     Experimental throughput analysis             " << '\n';
    std::cout << "      - A100 GPU CUDA Kernels                     " << '\n';
    std::cout << "      - C++17                                     " << '\n';
    std::cout << "     Project by: Jesse Kim                        " << '\n';
    std::cout << "==================================================" << '\n';

    return 0;
}