#ifndef COALESCED_MATMUL_CUH
#define COALESCED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        COALESCED GEMM KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for the final steps of optimization of
    a matrix multiplication kernel without using tensor cores.

    One of the most important optimizations for any GPU kernel
    is ensuring coalesced memory access as this enables DRAM
    bursting. Matrix multiplication reads for matrix B are coalesced
    by default since adjacent threads acess adjacent columns which, 
    by row-major order, are adjacent in DRAM. Adjacent threads for
    matrix A read from adjacent rows, which are not coalesced. We
    can optimize a standard kernel using corner tuning, which forces
    the irregular access pattern from memory to the shared memory,
    which has relatively lower latency compared to global memory.

    One must also note that the thread coarsening also has an impact 
    on memory access strucutre. In this way, it can be helpful
    to decouple the memory access and dot product assignments for
    coarser thread granularity. 

    The general process I used to optimize the memory access of this 
    kernel was decoupling the reads from shared memory and FLOPs from 
    the reads from global memory and storing into shared memory. In
    this way a coarser thread granularity no longer creates memeory
    access overhead and corner tuning can more naturally be applied to 
    matrix A. Each one of the loops essentially figures out the tile
    of A or B that it is reponsible for loading, then uses linearized
    thread ids to access those tiles in a coalesced manner.

*/

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void coalescedMatmulKernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    const int M, 
    const int N,
    const int K
) {
    int tx{static_cast<int>(threadIdx.x)};
    int ty{static_cast<int>(threadIdx.y)};
    int bx{static_cast<int>(blockIdx.x)};
    int by{static_cast<int>(blockIdx.y)};

    constexpr int TILE_WIDTH{BLOCK_SIZE * COARSE_FACTOR};
    constexpr int NUM_THREADS{BLOCK_SIZE * BLOCK_SIZE};

    int row_start{TILE_WIDTH * by + ty * COARSE_FACTOR};
    int col_start{TILE_WIDTH * bx + tx * COARSE_FACTOR};

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH + 1];

    float sums[COARSE_FACTOR*COARSE_FACTOR]{0.0f};

    int row_current{};
    int col_current{};

    int linear_tid{ty * BLOCK_SIZE + tx};

    int global_col_A{};
    int global_row_A{};
    int global_col_B{};
    int global_row_B{};

    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {

        int local_row{linear_tid / TILE_WIDTH};
        int local_col{linear_tid % TILE_WIDTH};

        #pragma unroll
        for(int load_idx{}; load_idx < COARSE_FACTOR*COARSE_FACTOR; ++load_idx) {

            global_col_A = tile_idx * TILE_WIDTH + local_col;
            global_row_A = by * TILE_WIDTH + local_row;

            global_col_B = bx * TILE_WIDTH + local_col;
            global_row_B = tile_idx * TILE_WIDTH + local_row;

            if(global_row_A < M && global_col_A < K)
                Ads[local_row][local_col] = A[global_row_A * K + global_col_A];
            else
                Ads[local_row][local_col] = 0.0f;

            if(global_row_B < K && global_col_B < N)
                Bds[local_row][local_col] = B[global_row_B * N + global_col_B];
            else
                Bds[local_row][local_col] = 0.0f;

            local_col += NUM_THREADS;
            while (local_col >= TILE_WIDTH) {
                local_col -= TILE_WIDTH;
                ++local_row;
            }
        }

        __syncthreads();

        #pragma unroll
        for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
            #pragma unroll
            for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
                #pragma unroll
                for(int i{}; i < TILE_WIDTH; ++i) {
                    sums[row_idx * COARSE_FACTOR + col_idx] += Ads[ty * COARSE_FACTOR + row_idx][i] * Bds[i][tx * COARSE_FACTOR + col_idx];
                }
            }
        }

        __syncthreads();

    }

    #pragma unroll
    for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
        #pragma unroll
        for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
            
            row_current = row_start + row_idx;
            col_current = col_start + col_idx;

            if((row_current < M) && (col_current < N))
                C[row_current * N + col_current] = sums[row_idx * COARSE_FACTOR + col_idx];

        }
    }
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
void coalescedMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    const int M, 
    const int N,
    const int K
) {

    constexpr int TILE_WIDTH = BLOCK_SIZE * COARSE_FACTOR;

    dim3 dimBlock(
        BLOCK_SIZE, 
        BLOCK_SIZE, 
        1
    );
    dim3 dimGrid(
        (N + TILE_WIDTH - 1)/TILE_WIDTH,
        (M + TILE_WIDTH - 1)/TILE_WIDTH,
        1
    );

    coalescedMatmulKernel<BLOCK_SIZE, COARSE_FACTOR><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Coalexed Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';

}

#endif