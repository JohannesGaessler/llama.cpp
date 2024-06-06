#include "quantize.cuh"
#include <cstdint>

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx_padded) {
    const int64_t ix = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int64_t iy = (int64_t)blockDim.y*blockIdx.y + threadIdx.y;

    const int64_t i_padded = (int64_t)iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

static __global__ void quantize_mmq_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx_padded) {
    const int64_t ix = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int64_t iy = (int64_t)blockDim.y*blockIdx.y + threadIdx.y;

    const int64_t i_padded = (int64_t)iy*kx_padded + ix;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib = i_padded / (4*QK8_1); // block index
    const int64_t iqs = i_padded % (4*QK8_1); // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs % QK8_1 != 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds[iqs/QK8_1].x) = d;
    reinterpret_cast<half&>(y[ib].ds[iqs/QK8_1].y) = sum;
}

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream) {
    GGML_ASSERT(kx_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

void quantize_mmq_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream) {
    GGML_ASSERT(kx_padded % (4*QK8_1) == 0);

    static_assert(CUDA_QUANTIZE_MMQ_BLOCK_SIZE == 4*QK8_1, "Block sizes != 4*QK8_1 == 128 not implemented.");
    const int64_t block_num_x = (kx_padded + CUDA_QUANTIZE_MMQ_BLOCK_SIZE - 1) / CUDA_QUANTIZE_MMQ_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_mmq_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}
