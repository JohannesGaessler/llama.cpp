#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f32.cuh"

static int fattn_tile_get_kq_stride_host(const int D, const int ncols, const int warp_size) {
    switch (D) {
        case 64:
        case 128:
        case 256:
            return ncols <= 16 || warp_size == 64 ? 64 : 32;
        default:
            GGML_ABORT("fatal error");
            return -1;
    }
}

static constexpr __device__ int fattn_tile_get_kq_stride_device(int D, int ncols) {
    switch (D) {
        case 64:
        case 128:
        case 256:
            return ncols <= 16 || ggml_cuda_get_physical_warp_size() == 64 ? 64 : 32;
        default:
            return -1;
    }
}

template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 2)
static __global__ void flash_attn_tile_ext_f32(
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

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int kq_stride = fattn_tile_get_kq_stride_device(D, ncols);
    static_assert(D         % (2*warp_size) == 0, "D not divisible by 2*warp_size.");
    static_assert(kq_stride %    warp_size  == 0, "kq_stride not divisable by warp_size.");
    constexpr int kq_nbatch = D == 128 ? 128 : 64;

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2   = (const float2 *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const half2  * K_h2   = (const half2  *) (K    + nb13* sequence         + nb12*(head / gqa_ratio));
    const half2  * V_h2   = (const half2  *) (V    + nb13* sequence         + nb12*(head / gqa_ratio)); // K and V have same shape
    const half   * maskh  = (const half   *) (mask  + nb33*(sequence % ne33)                          + nb31*ic0);
    const float  * sinksf = (const float  *) (sinks);

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    __shared__ float KQ[ncols][kq_stride];
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_sram[ncols][D/2];
    __shared__ half2 KQ[ncols][kq_stride/2];
    __shared__ half2 KV_tmp[kq_stride * (kq_nbatch/2 + 1)]; // Padded to avoid memory bank conflicts.
    half2 * KQ_h2 = (half2 *) KQ;
    half2 VKQ[ncols/nwarps][D/(2*warp_size)] = {{{0.0f, 0.0f}}};
#else
    __shared__ float Q_sram[ncols][D];
    __shared__ float KV_tmp[kq_stride * (kq_nbatch + 1)]; // Padded to avoid memory bank conflicts.
    float2 * KV_tmp2 = (float2 *) KV_tmp;
    float2 VKQ[ncols/nwarps][D/(2*warp_size)] = {{{0.0f, 0.0f}}};
#endif // FAST_FP16_AVAILABLE


    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += warp_size) {
            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i0 + threadIdx.x] : make_float2(0.0f, 0.0f);
#ifdef FAST_FP16_AVAILABLE
            Q_sram[j][i0 + threadIdx.x] = make_half2(tmp.x * scale, tmp.y * scale);
#else
            Q_sram[j][2*i0             + threadIdx.x] = tmp.x * scale;
            Q_sram[j][2*i0 + warp_size + threadIdx.x] = tmp.y * scale;
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

#pragma unroll
                for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch/2; k_KQ_1 += warp_size) {
                    const half2 tmph = K_h2[int64_t(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + k_KQ_1 + threadIdx.x];
#ifdef FAST_FP16_AVAILABLE
                    KV_tmp[i_KQ*(kq_nbatch + 1) + k_KQ_1 + threadIdx.x] = tmpf.x;
#else
                    const float2 tmpf = __half22float2(tmph);
                    KV_tmp[i_KQ*(kq_nbatch + 1) + 2*k_KQ_1             + threadIdx.x] = tmpf.x;
                    KV_tmp[i_KQ*(kq_nbatch + 1) + 2*k_KQ_1 + warp_size + threadIdx.x] = tmpf.y;
#endif // FAST_FP16_AVAILABLE
                }
            }

            __syncthreads();

#ifdef FAST_FP16_AVAILABLE
            constexpr int k_KQ_1_max = kq_nbatch/2;
#else
            constexpr int k_KQ_1_max = kq_nbatch;
#endif // FAST_FP16_AVAILABLE
#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < k_KQ_1_max; ++k_KQ_1) {
#ifdef FAST_FP16_AVAILABLE
                half2 K_k[kq_stride/warp_size];
                half2 Q_k[ncols/nwarps];
#else
                float K_k[kq_stride/warp_size];
                float Q_k[ncols/nwarps];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
                    const int i_KQ = i_KQ_0 + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
                    K_k[i_KQ_0/warp_size] = KV_tmp[i_KQ*(kq_nbatch/2 + 1) + k_KQ_1];
#else
                    K_k[i_KQ_0/warp_size] = KV_tmp[i_KQ*(kq_nbatch   + 1) + k_KQ_1];
#endif // FAST_FP16_AVAILABLE
                }
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    const int j_KQ = j_KQ_0 + threadIdx.y;

                    Q_k[j_KQ_0/nwarps] = Q_sram[j_KQ][k_KQ_0 + k_KQ_1];
                }

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
#ifdef FAST_FP16_AVAILABLE
                        const float2 tmp = __half22float2(K_k[i_KQ_0/warp_size] * Q_k[j_KQ_0/nwarps]);
                        sum[i_KQ_0/warp_size][j_KQ_0/nwarps] += tmp.x + tmp.y;
#else
                        sum[i_KQ_0/warp_size][j_KQ_0/nwarps] += K_k[i_KQ_0/warp_size] * Q_k[j_KQ_0/nwarps];
#endif // FAST_FP16_AVAILABLE
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
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                if (use_logit_softcap) {
                    sum[i_KQ_0/warp_size][j_KQ_0/nwarps] = logit_softcap * tanhf(sum[i_KQ_0/warp_size][j_KQ_0/nwarps]);
                }

                sum[i_KQ_0/warp_size][j_KQ_0/nwarps] += mask ? slope*__half2float(maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum[i_KQ_0/warp_size][j_KQ_0/nwarps]);

                KQ[j_KQ][i_KQ] = sum[i_KQ_0/warp_size][j_KQ_0/nwarps];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max<warp_size>(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < kq_stride; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const float diff = KQ[j][i] - kqmax[j0/nwarps];
                const float val = expf(diff);
                kqsum_add += val;
                KQ[j][i] = val;
            }
            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + kqsum_add;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0/nwarps][i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0/nwarps][i0/warp_size].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/warp_size].y *= KQ_max_scale;
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

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;

                    const half2 tmp = V_h2[int64_t(k_VKQ_0 + k0 + k_tile)*stride_KV2 + i];
#ifdef FAST_FP16_AVAILABLE
                    KV_tmp[k_tile*(D/2) + i] = tmp;
#else
                    KV_tmp2[k_tile*(D/2) + i] = __half22float2(tmp);
#endif // FAST_FP16_AVAILABLE
                }
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

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
                    V_k[i0/warp_size] = KV_tmp[k1*(D/2) + i];
#else
                    V_k[i0/warp_size] = KV_tmp2[k1*(D/2) + i];
#endif // FAST_FP16_AVAILABLE
                }
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    const int j = j0 + threadIdx.y;

#ifdef FAST_FP16_AVAILABLE
                    const float tmp = KQ[j][k0 + k1];
                    KQ_k[j0/nwarps] = make_half2(tmp, tmp);
#else
                    KQ_k[j0/nwarps] = KQ[j][k0 + k1];
#endif // FAST_FP16_AVAILABLE
                }

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
#pragma unroll
                    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
#ifdef FAST_FP16_AVAILABLE
                        VKQ[j0/nwarps][i0/warp_size]   += V_k[i0/warp_size]  *KQ_k[j0/nwarps];
#else
                        VKQ[j0/nwarps][i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[j0/nwarps];
                        VKQ[j0/nwarps][i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[j0/nwarps];
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
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            float kqmax_new_j = fmaxf(kqmax[j0/nwarps], sink);
            kqmax_new_j = warp_reduce_max<warp_size>(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new_j);
            kqmax[j0/nwarps] = kqmax_new_j;

            const float val = expf(sink - kqmax[j0/nwarps]);
            kqsum[j0/nwarps] = kqsum[j0/nwarps] * KQ_max_scale;
            if (threadIdx.x == 0) {
                kqsum[j0/nwarps] += val;
            }

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0/nwarps][i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0/nwarps][i0/warp_size].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    float2 * dst2 = (float2 *) dst;

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0/nwarps];
        kqsum_j = warp_reduce_sum<warp_size>(kqsum_j);

        const int j_dst_unrolled = ((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;

#pragma unroll
        for (int i00 = 0; i00 < D/2; i00 += warp_size) {
            const int i0 = i00 + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
            float2 dst_val = __half22float2(VKQ[j_VKQ_0/nwarps][i0/warp_size]);
#else
            float2 dst_val = VKQ[j_VKQ_0/nwarps][i0/warp_size];
#endif // FAST_FP16_AVAILABLE
            if (gridDim.y == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            dst2[j_dst_unrolled*(D/2) + i0] = dst_val;
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
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
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    constexpr int    nwarps        = 8;
    constexpr size_t nbytes_shared = 0;

    if (Q->ne[1] > 16) {
        constexpr int cols_per_block = 32;
        fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
        const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, warp_size);
        launch_fattn<D, cols_per_block, 1>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
        return;
    }

    constexpr int cols_per_block = 16;
    fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
    const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, warp_size);
    launch_fattn<D, cols_per_block, 1>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
}

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        switch (Q->ne[0]) {
#ifndef GGML_USE_HIP
            // FIXME
            case  64:
                launch_fattn_tile_switch_ncols< 64, use_logit_softcap>(ctx, dst);
                break;
#endif // GGML_USE_HIP
            case 128:
                launch_fattn_tile_switch_ncols<128, use_logit_softcap>(ctx, dst);
                break;
            case 256:
                launch_fattn_tile_switch_ncols<256, use_logit_softcap>(ctx, dst);
                break;
            default:
                GGML_ABORT("Unsupported head size");
                break;
        }
    } else {
        constexpr bool use_logit_softcap = true;
        switch (Q->ne[0]) {
            case 128:
                launch_fattn_tile_switch_ncols<128, use_logit_softcap>(ctx, dst);
                break;
            case 256:
                launch_fattn_tile_switch_ncols<256, use_logit_softcap>(ctx, dst);
                break;
            default:
                GGML_ABORT("Unsupported head size");
                break;
        }
    }
}
