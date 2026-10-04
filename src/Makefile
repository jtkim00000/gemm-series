# ============================================================
# CUDA GEMM Kernel Optimization Project
# ============================================================

NVCC = /usr/bin/nvcc

TARGET = gemm_benchmark

# RTX 4060 Laptop = Compute Capability 8.9
CUDA_ARCH = -arch=sm_89

NVCCFLAGS = -O3 -std=c++17 $(CUDA_ARCH) -Xcompiler -Wall

# CUDA 12.4 cuBLAS
LIBS = -L/usr/lib/x86_64-linux-gnu -lcublas

SRC = main.cu
OBJ = main.o

# ============================================================
# Build
# ============================================================

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) $(LIBS) -o $(TARGET)

main.o: main.cu \
	benchmark/benchmark.cuh \
	matmul/naive_matmul.cuh \
	matmul/tiled_matmul.cuh \
	matmul/thread_coarsened_matmul.cuh \
	matmul/coalesced_matmul.cuh \
	matmul/peak_matmul.cuh

	$(NVCC) $(NVCCFLAGS) -c main.cu -o main.o

# ============================================================
# Run
# ============================================================

run: $(TARGET)
	./$(TARGET)

# ============================================================
# Clean
# ============================================================

clean:
	rm -f $(OBJ) $(TARGET)

rebuild: clean all

.PHONY: all run clean rebuild