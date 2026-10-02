#ifndef THREAD_COARSENED_MATMUL_CUH
#define THREAD_COARSENED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        2D THREAD COARSENED GEMM KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for a 2D thread coarsened matrix multiplication
    kernel. The primary advantage of parallelizing work across 
    threads at the finest possible granularity is that it maximizes
    parallelism and enhaces scalability. However, there can often
    be drawbacks to this fine granularity. For exmaple, redundant 
    data loads by different blocks, synchronization overhead etc.
    Oftentimes it is benefitial to partially serialize the work 
    done by threads to reduce the potential overheads from fine
    granularity. This is done by assigning more work for each 
    thread. This is the essence of thread coarsening.

    The tiled matrix multiplication kernel in tile_matmul.cuh can 
    be modified to use the thread coarsening optimization. In this
    case the thread coarsening is applied to both directions. It is
    important to note that 2D thread coarsening can cause complicates
    as it has more complex indexing, the register usage grows quadratically,
    and it is much more hardware-sensitive due to potential impacts
    on occupancy. However it causes both operands to get reused, and it 
    maximizes FLOPs-per-shared-memory-access.

    In this case each thread is computing a square of width 
    COARSE_FACTOR in the output matrix. This primary reduces
    the overhead from thread synchronization and data redundancy
    across blocks. One must be careful not to choose too large
    of a COARSE_FACTOR as it can cause underutlization of parallel
    computing reasources. 

    An important change between the tiled matrix multiplication
    kernel and a thread coarsened one is that the block and tile
    size are no longer the same. Instead the tile size is a product
    of the blocksize and the coarsening factor. With templates this
    can be constexpr. 
*/

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void threadCoarsenedMatmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    const int M, 
    const int N,
    const int K
) {
    int tx{static_cast<int>(threadIdx.x)};
    int ty{static_cast<int>(threadIdx.y)};
    int bx{static_cast<int>(blockIdx.x)};
    int by{static_cast<int>(blockIdx.y)};

    constexpr int TILE_WIDTH{BLOCK_SIZE * COARSE_FACTOR};

    int row_start{TILE_WIDTH * by + ty * COARSE_FACTOR};
    int col_start{TILE_WIDTH * bx + tx * COARSE_FACTOR};

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    float sums[COARSE_FACTOR*COARSE_FACTOR];  // row-major output sums

    for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
        for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
            sums[row_idx * COARSE_FACTOR + col_idx] = 0.0f;
        }
    }
    
    int row_current{};
    int col_current{};

    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {

        for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
            for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {

                row_current = row_start + row_idx;
                col_current = col_start + col_idx;

                if((row_current < M) && ((tile_idx * TILE_WIDTH + tx * COARSE_FACTOR + col_idx) < K))
                    Ads[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = A[row_current * K + tile_idx * TILE_WIDTH + tx * COARSE_FACTOR + col_idx];
                else
                    Ads[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = 0.0f;

                if((col_current < N) && ((tile_idx * TILE_WIDTH + ty * COARSE_FACTOR + row_idx) < K))
                    Bds[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = B[(N * (tile_idx * TILE_WIDTH + ty * COARSE_FACTOR + row_idx)) + col_current];
                else
                    Bds[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = 0.0f;
            }
        }

        __syncthreads();


        for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
            for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {

                for(int i{}; i < TILE_WIDTH; ++i) {
                    sums[row_idx * COARSE_FACTOR + col_idx] += Ads[ty * COARSE_FACTOR + row_idx][i] * Bds[i][tx * COARSE_FACTOR + col_idx];
                }
            }
        }

        __syncthreads();

    }

    for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
        for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
            
            row_current = row_start + row_idx;
            col_current = col_start + col_idx;

            if((row_current < M) && (col_current < N))
                C[row_current * N + col_current] = sums[row_idx * COARSE_FACTOR + col_idx];

        }
    }
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
void threadCoarsenedMatmulGPU(
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
        (M + TILE_WIDTH - 1)/TILE_WIDTH,
        (N + TILE_WIDTH - 1)/TILE_WIDTH,
        1
    );

    threadCoarsenedMatmulKernel<BLOCK_SIZE, COARSE_FACTOR><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Thread Coarsened Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
        std::cout << "Thread Coarsened Matmul Kernel Exectution Error: " << cudaGetErrorString(err) << '\n';
}

#endif