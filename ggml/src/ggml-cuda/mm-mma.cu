#include "ggml.h"
#include "common.cuh"
#include "mma.cuh"
#include "mm-mma.cuh"

using namespace ggml_cuda_mma;

template <typename T, int rows_per_block, int cols_per_block, int nwarps>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
static __global__ void mul_mat_f(
        const T * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols2, const int nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    typedef tile<16, 8, T>     tile_A;
    typedef tile< 8, 8, T>     tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int tile_k_padded = warp_size + 4;
    constexpr int ntA = rows_per_block / tile_A::I;
    constexpr int ntB = cols_per_block / tile_B::I;

    const int row0        = blockIdx.x * rows_per_block;
    const int channel_dst = blockIdx.y;
    const int channel_x   = channel_dst / channel_ratio;
    const int channel_y   = channel_dst;
    const int sample_dst  = blockIdx.z;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row0*stride_row ;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];

    tile_C C[ntA][ntB];

    if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
        T * tile_xy = (T *) data_mmv + threadIdx.y*(tile_A::I * tile_k_padded);

        for (int col2 = threadIdx.y*warp_size + threadIdx.x; col2 < ncols2; col2 += nwarps*warp_size) {
            tile_A A[ntA][warp_size / tile_A::J];
#pragma unroll
            for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
                for (int i = 0; i < tile_A::I; ++i) {
                    tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col2];
                }
#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                    load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
                }
            }

#pragma unroll
            for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
                for (int j = 0; j < tile_B::I; ++j) {
                    const float2 tmp = y2[(itB*tile_B::I + j)*stride_col_y2 + col2];
                    tile_xy[j*tile_k_padded + threadIdx.x] = {tmp.x, tmp.y};
                }
#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                    tile_B B;
                    load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                    for (int itA = 0; itA < ntA; ++itA) {
                        mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                    }
                }
            }
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    float * buf_iw = (float *) data_mmv;
    constexpr int kiw = nwarps*rows_per_block + 4;

    if (nwarps > 1) {
        __syncthreads();
    }
#pragma unroll
    for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = threadIdx.y*rows_per_block + itA*tile_C::I + tile_C::get_i(l);
                const int j = itB*tile_C::J + tile_C::get_j(l);
                buf_iw[j*kiw + i] = C[itA][itB].x[l];
            }
        }
    }

    if (nwarps > 1) {
        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j0 + nwarps > cols_per_block && j >= cols_per_block) {
            return;
        }

        float sum = 0.0f;
        static_assert(rows_per_block == warp_size, "need loop/check");
#pragma unroll
        for (int i0 = 0; i0 < nwarps*rows_per_block; i0 += rows_per_block) {
            const int i = i0 + threadIdx.x;

            sum += buf_iw[j*kiw + i];
        }
        dst[j*stride_col_dst + row0 + threadIdx.x] = sum;
    }
}

template <typename T, int cols_per_block>
static void mul_mat_mma_cuda(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream) {
    typedef tile<16, 8, T>     tile_A;
    typedef tile< 8, 8, T>     tile_B;
    typedef tile<16, 8, float> tile_C;

    GGML_ASSERT(ncols_x      % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const int64_t channel_ratio = nchannels_dst / nchannels_x;
    const int64_t sample_ratio  = nsamples_dst  / nsamples_x;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t nwarps_best     = 1;
    int64_t niter_best      = (ncols_x + warp_size*2 - 1) / (warp_size*2);
    int64_t max_block_size  = 256;
    for (int64_t nwarps = 2; nwarps <= max_block_size/warp_size; nwarps++) {
        const int64_t niter = (ncols_x + nwarps*warp_size*2 - 1) / (nwarps*warp_size*2);
        if (niter < niter_best) {
            niter_best  = niter;
            nwarps_best = nwarps;
        }
    }

    constexpr int rows_per_block = 32;
    const int nbytes_shared_iter = nwarps_best * tile_A::I * (warp_size + 4) * 4;
    const int nbytes_shared_combine = cols_per_block * (nwarps_best*rows_per_block + 4) * 4;
    const int nbytes_shared = std::max(nbytes_shared_iter, nbytes_shared_combine);
    const dim3 block_nums(nrows_x/rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, nwarps_best, 1);
    switch (nwarps_best) {
        case 1: {
            mul_mat_f<T, rows_per_block, cols_per_block, 1><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 2: {
            mul_mat_f<T, rows_per_block, cols_per_block, 2><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 3: {
            mul_mat_f<T, rows_per_block, cols_per_block, 3><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 4: {
            mul_mat_f<T, rows_per_block, cols_per_block, 4><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 5: {
            mul_mat_f<T, rows_per_block, cols_per_block, 5><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 6: {
            mul_mat_f<T, rows_per_block, cols_per_block, 6><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 7: {
            mul_mat_f<T, rows_per_block, cols_per_block, 7><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 8: {
            mul_mat_f<T, rows_per_block, cols_per_block, 8><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, y, ids, dst, ncols_x, nchannels_y, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template <typename T>
static void mul_mat_f_switch_cols_per_block(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream) {
    if (ncols_dst <= 8) {
        mul_mat_mma_cuda<T, 8>(x, y, ids, dst, ncols_x, nrows_x, stride_row, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_x,                nsamples_dst,  stride_sample_x,  stride_sample_y,  stride_sample_dst,  stream);
        return;
    }
    GGML_ASSERT(ncols_dst <= 16);
    mul_mat_mma_cuda<T, 16>(x, y, ids, dst, ncols_x, nrows_x, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_x,                nsamples_dst,  stride_sample_x,  stride_sample_y,  stride_sample_dst,  stream);
    return;
}

void ggml_cuda_mul_mat_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    GGML_ASSERT(!ids || ncols_dst == 1);

    switch (src0->type) {
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_cols_per_block(src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, s11, s1,
                ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03/vals_per_T, s13,              s3,                 ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_cols_per_block(src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, s11, s1,
                ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03/vals_per_T, s13,              s3,                 ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
