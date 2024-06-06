#pragma once

#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE 256

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);

struct block_q8_1_mmq {
    half2  ds[4];
    int8_t qs[4*QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);
