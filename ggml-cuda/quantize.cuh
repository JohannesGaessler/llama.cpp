#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE 256

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);
