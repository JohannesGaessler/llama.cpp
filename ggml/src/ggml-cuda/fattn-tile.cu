#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile.cuh"

#define FATTN_TILE_NTHREADS 256

static int fattn_tile_get_kq_stride_host(const int D, const int ncols, const int cc, const int warp_size) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        switch (D) {
            case 64:
                return 64;
            case 128:
            case 256:
                if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
                    return ncols <= 16 ? 64 : 32;
                } else {
                    return 64;
                }
            default:
                GGML_ABORT("fatal error");
                return -1;
        }
    }
    if (fast_fp16_available(cc)) {
        switch (D) {
            case 64:
            case 128:
                return 128;
            case 256:
                return ncols <= 16 ? 128 : 64;
            default:
                GGML_ABORT("fatal error");
                return -1;
        }
    }
    switch (D) {
        case 64:
            return ncols <= 16 ? 128 : 64;
        case 128:
            return ncols <= 16 ? 64 : 32;
        case 256:
            return 32;
        default:
            GGML_ABORT("fatal error");
            return -1;
    }
    GGML_UNUSED(warp_size);
}

static constexpr __device__ int fattn_tile_get_kq_stride_device(int D, int ncols, int warp_size) {
#ifdef GGML_USE_HIP
    switch (D) {
        case 64:
            return 64;
        case 128:
#if defined(GCN) || defined(CDNA)
            return ncols <= 16 ? 64 : 32;
#else
            return 64;
#endif // defined(GCN) || defined(CDNA)
        case 256:
#if defined(GCN) || defined(CDNA)
            return ncols <= 16 ? 64 : 32;
#else
            return 64;
#endif // defined(GCN) || defined(CDNA)
        default:
            return -1;
    }
#else
#ifdef FAST_FP16_AVAILABLE
    switch (D) {
        case 64:
        case 128:
            return 128;
        case 256:
            return ncols <= 16 ? 128 : 64;
        default:
            return -1;
    }
#else
    switch (D) {
        case 64:
            return ncols <= 16 ? 128 : 64;
        case 128:
            return ncols <= 16 ? 64 : 32;
        case 256:
            return 32;
        default:
            return -1;
    }
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
    GGML_UNUSED_VARS(ncols, warp_size);
}

static constexpr __device__ int fattn_tile_get_kq_nbatch_device(int D, int ncols, int warp_size) {
#ifdef GGML_USE_HIP
    switch (D) {
        case 64:
            return 64;
        case 128:
#if defined(GCN) || defined(CDNA)
            return ncols <= 16 ? 64 : 128;
#else
            return 64;
#endif // defined(GCN) || defined(CDNA)
        case 256:
#if defined(GCN) || defined(CDNA)
            return ncols <= 16 ? 64 : 128;
#else
            return ncols <= 16 ? 64 : 256;
#endif // defined(GCN) || defined(CDNA)
        default:
            return -1;
    }
#else
#ifdef FAST_FP16_AVAILABLE
    switch (D) {
        case 64:
            return 64;
        case 128:
            return ncols <= 16 ? 128 : 64;
        case 256:
            return ncols <= 16 ? 64 : 128;
        default:
            return -1;
    }
#else
    switch (D) {
        case 64:
            return 64;
        case 128:
            return 128;
        case 256:
            return ncols <= 16 ? 128 : 64;
        default:
            return -1;
    }
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
    GGML_UNUSED_VARS(ncols, warp_size);
}

template<int D, int ncols, bool use_logit_softcap> // D == head size
#ifdef GGML_USE_HIP
__launch_bounds__(FATTN_TILE_NTHREADS, 2)
#else
__launch_bounds__(FATTN_TILE_NTHREADS, 2)
#endif // GGML_USE_HIP
static __global__ void flash_attn_tile(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE

    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = 32;
    constexpr int nwarps    = FATTN_TILE_NTHREADS / warp_size;
    constexpr int kq_stride = fattn_tile_get_kq_stride_device(D, ncols, warp_size);
    static_assert(kq_stride % warp_size == 0, "kq_stride not divisable by warp_size.");
    constexpr int kq_nbatch = fattn_tile_get_kq_nbatch_device(D, ncols, warp_size);
    static_assert(kq_nbatch % (2*warp_size) == 0, "bad kq_nbatch");

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f    = (const float *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const half2 * K_h2   = (const half2 *) (K    + nb13* sequence         + nb12*(head / gqa_ratio));
    const half2 * V_h2   = (const half2 *) (V    + nb13* sequence         + nb12*(head / gqa_ratio)); // K and V have same shape
    const half  * maskh  = (const half  *) (mask  + nb33*(sequence % ne33)                          + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

#if defined(GGML_USE_HIP)
    constexpr int cpy_nb = 16;
#else
    constexpr int cpy_nb = 8;
#endif // defined(GGML_USE_HIP) && defined(GCN)
    constexpr int cpy_ne = cpy_nb / 4;

    __shared__ float KQ[ncols][kq_stride];
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_tmp[ncols][D/2];
    __shared__ half2 KV_tmp_h2[kq_stride * (kq_nbatch/2 + cpy_ne)]; // Padded to avoid memory bank conflicts.
    half2 VKQ[ncols/nwarps][D/(2*warp_size)] = {{{0.0f, 0.0f}}};
#else
    __shared__ float Q_tmp[ncols][D];
    __shared__ float KV_tmp_f[kq_stride * (kq_nbatch + cpy_ne)]; // Padded to avoid memory bank conflicts.
    float2 VKQ[ncols/nwarps][D/(2*warp_size)] = {{{0.0f, 0.0f}}};
#endif // FAST_FP16_AVAILABLE


    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j0 = 0; j0 < ncols/nwarps; ++j0) {
        const int j = j0 + threadIdx.y*(ncols/nwarps);

        constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
            float tmp_f[cpy_ne_D] = {0.0f};
            if (ic0 + j < ne01) {
                ggml_cuda_memcpy_1<sizeof(tmp_f)>(tmp_f, &Q_f[j*(nb01/sizeof(float)) + i0 + threadIdx.x*cpy_ne_D]);
            }

#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp_f[i1] *= scale;
            }

#ifdef FAST_FP16_AVAILABLE
            half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
            }
            ggml_cuda_memcpy_1<sizeof(tmp_h2)>(&Q_tmp[j][i0/2 + threadIdx.x*(cpy_ne_D/2)], tmp_h2);
#else
            ggml_cuda_memcpy_1<sizeof(tmp_f)> (&Q_tmp[j][i0   + threadIdx.x* cpy_ne_D],    tmp_f);
#endif // FAST_FP16_AVAILABLE
        }
    }

    __syncthreads();

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k_VKQ_0 = blockIdx.y*kq_stride; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*kq_stride) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        float sum[kq_stride/warp_size][ncols/nwarps] = {{0.0f}};

#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += kq_nbatch) {
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += nwarps) {
                const int i_KQ = i_KQ_0 + threadIdx.y;

#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_kqnb = cpy_ne < kq_nbatch/(2*warp_size) ? cpy_ne : kq_nbatch/(2*warp_size);
#pragma unroll
                for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch/2; k_KQ_1 += warp_size*cpy_ne_kqnb) {
                    ggml_cuda_memcpy_1<cpy_ne_kqnb*4>(
                        &KV_tmp_h2[i_KQ*(kq_nbatch/2 + cpy_ne) + k_KQ_1 + threadIdx.x*cpy_ne_kqnb],
                        &K_h2[int64_t(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + k_KQ_1 + threadIdx.x*cpy_ne_kqnb]);
                }
#else
                constexpr int cpy_ne_kqnb = cpy_ne < kq_nbatch/warp_size ? cpy_ne : kq_nbatch/warp_size;
#pragma unroll
                for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch; k_KQ_1 += warp_size*cpy_ne_kqnb) {
                    half2 tmp_h2[cpy_ne_kqnb/2];
                    ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, &K_h2[int64_t(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + k_KQ_1/2 + threadIdx.x*(cpy_ne_kqnb/2)]);

                    float2 tmp_f2[cpy_ne_kqnb/2];
#pragma unroll
                    for (int k_KQ_2 = 0; k_KQ_2 < cpy_ne_kqnb/2; ++k_KQ_2) {
                        tmp_f2[k_KQ_2] = __half22float2(tmp_h2[k_KQ_2]);
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f2)>(
                        &KV_tmp_f[i_KQ*(kq_nbatch + cpy_ne) + k_KQ_1 + threadIdx.x*cpy_ne_kqnb], tmp_f2);
                }
#endif // FAST_FP16_AVAILABLE
            }

            __syncthreads();

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch/2; k_KQ_1 += cpy_ne) {
                half2 K_k[kq_stride/warp_size][cpy_ne];
                half2 Q_k[ncols/nwarps][cpy_ne];
#else
#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch; k_KQ_1 += cpy_ne) {
                float K_k[kq_stride/warp_size][cpy_ne];
                float Q_k[ncols/nwarps][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
                    const int i_KQ = i_KQ_0 + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
                    ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/warp_size], &KV_tmp_h2[i_KQ*(kq_nbatch/2 + cpy_ne) + k_KQ_1]);
#else
                    ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/warp_size], &KV_tmp_f [i_KQ*(kq_nbatch   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
                }
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols/nwarps; ++j_KQ_0) {
                    const int j_KQ = j_KQ_0 + threadIdx.y*(ncols/nwarps);

#ifdef FAST_FP16_AVAILABLE
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_k[j_KQ_0], &Q_tmp[j_KQ][k_KQ_0/2 + k_KQ_1]);
#else
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_k[j_KQ_0], &Q_tmp[j_KQ][k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
                }

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < ncols/nwarps; ++j_KQ_0) {
#pragma unroll
                        for (int k = 0; k < cpy_ne; ++k) {
                            ggml_cuda_mad(sum[i_KQ_0/warp_size][j_KQ_0], K_k[i_KQ_0/warp_size][k], Q_k[j_KQ_0][k]);
                        }
                    }
                }
            }

            if (k_KQ_0 + kq_nbatch < D) {
                __syncthreads(); // Sync not needed on last iteration.
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols/nwarps; ++j_KQ_0) {
                const int j_KQ = j_KQ_0 + threadIdx.y*(ncols/nwarps);

                if (use_logit_softcap) {
                    sum[i_KQ_0/warp_size][j_KQ_0] = logit_softcap * tanhf(sum[i_KQ_0/warp_size][j_KQ_0]);
                }

                sum[i_KQ_0/warp_size][j_KQ_0] += mask ? slope*__half2float(maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0] = fmaxf(kqmax_new[j_KQ_0], sum[i_KQ_0/warp_size][j_KQ_0]);

                KQ[j_KQ][i_KQ] = sum[i_KQ_0/warp_size][j_KQ_0];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols/nwarps; ++j0) {
            const int j = j0 + threadIdx.y*(ncols/nwarps);

            kqmax_new[j0] = warp_reduce_max<warp_size>(kqmax_new[j0]);
            const float KQ_max_scale = expf(kqmax[j0] - kqmax_new[j0]);
            kqmax[j0] = kqmax_new[j0];

            float kqsum_add = 0.0f;
            if (kq_stride % (4*warp_size) == 0 && cpy_ne % 4 == 0) {
#pragma unroll
                for (int i0 = 0; i0 < kq_stride; i0 += 4*warp_size) {
                    const int i = i0 + 4*threadIdx.x;

                    float4 val = *(const float4 *) &KQ[j][i];
                    val.x = expf(val.x - kqmax[j0]);
                    val.y = expf(val.y - kqmax[j0]);
                    val.z = expf(val.z - kqmax[j0]);
                    val.w = expf(val.w - kqmax[j0]);
                    kqsum_add += val.x + val.y + val.z + val.w;

#ifdef FAST_FP16_AVAILABLE
                    const half2 tmp[2] = {make_half2(val.x, val.y), make_half2(val.z, val.w)};
                    ggml_cuda_memcpy_1<sizeof(tmp)>(&KQ[j][i/2], &tmp);
#else
                    ggml_cuda_memcpy_1<sizeof(val)>(&KQ[j][i], &val);
#endif // FAST_FP16_AVAILABLE
                }
            } else if (kq_stride % (2*warp_size) == 0 && cpy_ne % 2 == 0) {
#pragma unroll
                for (int i0 = 0; i0 < kq_stride; i0 += 2*warp_size) {
                    const int i = i0 + 2*threadIdx.x;

                    float2 val = *(const float2 *) &KQ[j][i];
                    val.x = expf(val.x - kqmax[j0]);
                    val.y = expf(val.y - kqmax[j0]);
                    kqsum_add += val.x + val.y;
#ifdef FAST_FP16_AVAILABLE
                    const half2 tmp = make_half2(val.x, val.y);
                    ggml_cuda_memcpy_1<sizeof(tmp)>(&KQ[j][i/2], &tmp);
#else
                    ggml_cuda_memcpy_1<sizeof(val)>(&KQ[j][i], &val);
#endif // FAST_FP16_AVAILABLE
                }
            } else {
                for (int i0 = 0; i0 < kq_stride; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;

                    const float diff = KQ[j][i] - kqmax[j0];
                    const float val = expf(diff);
                    kqsum_add += val;
#ifdef FAST_FP16_AVAILABLE
                    ((half *) KQ[j])[i] = val;
#else
                    KQ[j][i] = val;
#endif // FAST_FP16_AVAILABLE
                }
            }
            kqsum[j0] = kqsum[j0]*KQ_max_scale + kqsum_add;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size].x *= KQ_max_scale;
                VKQ[j0][i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }

        constexpr int V_cols_per_iter = kq_stride*kq_nbatch / D;
        static_assert(kq_stride % V_cols_per_iter == 0, "bad V_cols_per_iter");
#pragma unroll
        for (int k0 = 0; k0 < kq_stride; k0 += V_cols_per_iter) {
#pragma unroll
            for (int k1 = 0; k1 < V_cols_per_iter; k1 += nwarps) {
                const int k_tile = k1 + threadIdx.y;

#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_D = cpy_ne < D/(2*warp_size) ? cpy_ne : D/(2*warp_size);
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(
                        &KV_tmp_h2[k_tile*(D/2) + i0 + threadIdx.x*cpy_ne_D],
                        &V_h2[int64_t(k_VKQ_0 + k0 + k_tile)*stride_KV2 + i0 + threadIdx.x*cpy_ne_D]);
                }
#else
                constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
                    half2 tmp_h2[cpy_ne_D/2];
                    ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, &V_h2[int64_t(k_VKQ_0 + k0 + k_tile)*stride_KV2 + i0/2 + threadIdx.x*(cpy_ne_D/2)]);

                    float2 tmp_f2[cpy_ne_D/2];
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                        tmp_f2[i1] = __half22float2(tmp_h2[i1]);
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f2)>(
                        &KV_tmp_f[k_tile*D + i0 + threadIdx.x*cpy_ne_D], tmp_f2);
                }
#endif // FAST_FP16_AVAILABLE
            }

            __syncthreads();

#pragma unroll
            for (int k1 = 0; k1 < V_cols_per_iter; ++k1) {
#ifdef FAST_FP16_AVAILABLE
                half2 V_k[(D/2)/warp_size];
                half2 KQ_k[ncols/nwarps];
#else
                float2 V_k[(D/2)/warp_size];
                float  KQ_k[ncols/nwarps];
#endif // FAST_FP16_AVAILABLE

#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_D = cpy_ne/2 < (D/2)/warp_size ? cpy_ne/2 : (D/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp_h2[k1*(D/2) + i0 + threadIdx.x*cpy_ne_D]);
                }
#else
                constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp_f[k1*D + i0 + threadIdx.x*cpy_ne_D]);
                }
#endif // FAST_FP16_AVAILABLE
#pragma unroll
                for (int j0 = 0; j0 < ncols/nwarps; ++j0) {
                    const int j = j0 + threadIdx.y*(ncols/nwarps);

#ifdef FAST_FP16_AVAILABLE
                    KQ_k[j0] = __half2half2(((const half *)KQ[j])[k0 + k1]);
#else
                    KQ_k[j0] = KQ[j][k0 + k1];
#endif // FAST_FP16_AVAILABLE
                }

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
#pragma unroll
                    for (int j0 = 0; j0 < ncols/nwarps; ++j0) {
#ifdef FAST_FP16_AVAILABLE
                        VKQ[j0][i0/warp_size]   += V_k[i0/warp_size]  *KQ_k[j0];
#else
                        VKQ[j0][i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[j0];
                        VKQ[j0][i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[j0];
#endif // FAST_FP16_AVAILABLE
                    }
                }
            }

            __syncthreads();
        }
    }


    // Attention sink: adjust running max and sum once per head
    if (sinksf && blockIdx.y == 0) {
        const float sink = sinksf[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols/nwarps; ++j0) {
            float kqmax_new_j = fmaxf(kqmax[j0], sink);
            kqmax_new_j = warp_reduce_max<warp_size>(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j0] - kqmax_new_j);
            kqmax[j0] = kqmax_new_j;

            const float val = expf(sink - kqmax[j0]);
            kqsum[j0] = kqsum[j0] * KQ_max_scale;
            if (threadIdx.x == 0) {
                kqsum[j0] += val;
            }

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size].x *= KQ_max_scale;
                VKQ[j0][i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols/nwarps; ++j_VKQ_0) {
        kqsum[j_VKQ_0] = warp_reduce_sum<warp_size>(kqsum[j_VKQ_0]);
    }
    if (gridDim.y == 1) {
#pragma unroll
        for (int j_VKQ_0 = 0; j_VKQ_0 < ncols/nwarps; ++j_VKQ_0) {
#ifdef FAST_FP16_AVAILABLE
            const half2 kqsum_j_inv = make_half2(1.0f/kqsum[j_VKQ_0], 1.0f/kqsum[j_VKQ_0]);
#pragma unroll
            for (int i = 0; i < (D/2)/warp_size; ++i) {
                VKQ[j_VKQ_0][i] *= kqsum_j_inv;
            }
#else
            const float kqsum_j_inv = 1.0f/kqsum[j_VKQ_0];
#pragma unroll
            for (int i = 0; i < (D/2)/warp_size; ++i) {
                VKQ[j_VKQ_0][i].x *= kqsum_j_inv;
                VKQ[j_VKQ_0][i].y *= kqsum_j_inv;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols/nwarps; ++j_VKQ_0) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y*(ncols/nwarps);

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        const int j_dst_unrolled = ((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (D/2)/warp_size ? cpy_ne/2 : (D/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
            float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[j_VKQ_0][i0/warp_size + i1]);
            }
            ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*D + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
        }
#else
        constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
            ggml_cuda_memcpy_1<cpy_ne_D*4>(
                &dst[j_dst_unrolled*D + i0 + threadIdx.x*cpy_ne_D], &VKQ[j_VKQ_0][i0/(2*warp_size)]);
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(kqmax[j_VKQ_0], kqsum[j_VKQ_0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    const int id                 = ggml_cuda_get_device();
    const int cc                 = ggml_cuda_info().devices[id].cc;
    const int warp_size          = 32;
    const int nwarps             = FATTN_TILE_NTHREADS / warp_size;

    constexpr size_t nbytes_shared = 0;

    if (Q->ne[1] > 16) {
        constexpr int cols_per_block = 32;
        fattn_kernel_t fattn_kernel = flash_attn_tile<D, cols_per_block, use_logit_softcap>;
        const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, cc, warp_size);
        launch_fattn<D, cols_per_block, 1>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
        return;
    }

    constexpr int cols_per_block = 16;
    fattn_kernel_t fattn_kernel = flash_attn_tile<D, cols_per_block, use_logit_softcap>;
    const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, cc, warp_size);
    launch_fattn<D, cols_per_block, 1>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
}

template <bool use_logit_softcap>
static void launch_fattn_tile_switch_head_size(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            launch_fattn_tile_switch_ncols< 64, use_logit_softcap>(ctx, dst);
        } break;
        case 128: {
            launch_fattn_tile_switch_ncols<128, use_logit_softcap>(ctx, dst);
        } break;
        case 256: {
            launch_fattn_tile_switch_ncols<256, use_logit_softcap>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_head_size<use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_head_size<use_logit_softcap>(ctx, dst);
    }
}
