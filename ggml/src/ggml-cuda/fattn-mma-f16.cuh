#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * __restrict__ Q_f2,
        const half2  * __restrict__ K_h2,
        const half2  * __restrict__ V_h2,
        const half   * __restrict__ maskh,
        float        * __restrict__ dstk,
        float2       * __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef NEW_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    typedef mma_A_I16K8<half2> mma_A;
    typedef mma_B_J8K8<half2>  mma_B;
    typedef mma_C_I16J8<float> mma_C_KQ;
    typedef mma_C_I16J8<half2> mma_C_VKQ;

    const int D2_padded = D/2 + 4;     // Size of D in half2, padded to avoid shared memory bank conflicts.
    extern __shared__ half2 tile_KV[]; // Temporary shared buffer for loading K/V data with KQ_stride*D logical elements.

    const int stride_Q    = nb01 / sizeof(float2);
    const int stride_KV   = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    mma_B Q_B[D/(2*mma_B::K)];
    mma_C_VKQ VKQ_C[D/mma_C_VKQ::I];

    float2    KQ_rowsum = {0.0f, 0.0f};
    float2       KQ_max = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float2 KQ_max_scale = {0.0f, 0.0f};

    // Temporarily load Q data into tile_KV, will be loaded into registers afterwards.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
        const int stride_j = WARP_SIZE / stride_k;

#pragma unroll
        for (int j0 = 0; j0 < mma_B::J; j0 += stride_j) {
            const int j = j0 + threadIdx.y*mma_B::J + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jt*ncols + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols + j)*stride_Q + k];
                    tile_KV[j*D2_padded + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[j*D2_padded + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    {
        const int j0 = threadIdx.y*mma_B::J;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += mma_B::K) {
            Q_B[k0/mma_B::K].load_ldmatrix(tile_KV + j0*D2_padded + k0, D2_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const int k_VKQ_0 = kb0*KQ_stride;
        mma_C_KQ KQ_C[KQ_stride/mma_C_KQ::I];

        // Load K data into tile with decreasing granularity for D for better memory bandwidth:
#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
            const int stride_i = WARP_SIZE/stride_k;

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < KQ_stride; i_KQ_0 += nwarps*stride_i) {
                const int i_KQ = i_KQ_0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += stride_k) {
                    const int k_KQ = k_KQ_0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i_KQ*D2_padded + k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV + k_KQ];
                }
            }
        }

        __syncthreads();

        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < KQ_stride; i_KQ_0 += mma_A::I) {
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += mma_A::K) {
                mma_A K_A;
                K_A.load_ldmatrix(tile_KV + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
                KQ_C[i_KQ_0/mma_A::I].mma(K_A, Q_B[k_KQ_0/mma_A::K]);
            }
        }

        __syncthreads();

        if (use_logit_softcap) {
#pragma unroll
            for (int i0 = 0; i0 < KQ_stride; i0 += mma_C_KQ::I) {
#pragma unroll
                for (int l = 0; l < mma_C_KQ::ne; ++l) {
                    KQ_C[i0/mma_C_KQ::I].x[l] = logit_softcap*tanhf(KQ_C[i0/mma_C_KQ::I].x[l]);
                }
            }
        }

        if (maskh) {
#pragma unroll
            for (int i0 = 0; i0 < KQ_stride; i0 += mma_C_KQ::I) {
#pragma unroll
                for (int l = 0; l < mma_C_KQ::ne; ++l) {
                    const int i = i0 + mma_C_KQ::get_i(l);
                    const int j = threadIdx.y*mma_C_KQ::J + mma_C_KQ::get_j(l);

                    KQ_C[i0/mma_C_KQ::I].x[l] += slope*__half2float(maskh[j*stride_mask + k_VKQ_0 + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        float2 KQ_max_new = KQ_max;
#pragma unroll
        for (int k0 = 0; k0 < KQ_stride; k0 += mma_C_KQ::I) {
#pragma unroll
            for (int l0 = 0; l0 < mma_C_KQ::ne; l0 += 2) {
                const int k = k0 + mma_C_KQ::get_i(l0);

                KQ_max_new.x = fmaxf(KQ_max_new.x, KQ_C[k0/mma_C_KQ::I].x[l0 + 0]);
                KQ_max_new.y = fmaxf(KQ_max_new.y, KQ_C[k0/mma_C_KQ::I].x[l0 + 1]);
            }
        }

        // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
        for (int offset = 16; offset > 2; offset >>= 1) {
            KQ_max_new.x = fmaxf(KQ_max_new.x, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.x, offset, WARP_SIZE));
            KQ_max_new.y = fmaxf(KQ_max_new.y, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.y, offset, WARP_SIZE));
        }

        {
            const float2 diff = make_float2(KQ_max.x - KQ_max_new.x, KQ_max.y - KQ_max_new.y);
            KQ_max_scale = make_float2(expf(diff.x), expf(diff.y));
            if (diff.x <= SOFTMAX_FTZ_THRESHOLD) {
                KQ_max_scale.x = 0.0f;
            }
            if (diff.y <= SOFTMAX_FTZ_THRESHOLD) {
                KQ_max_scale.y = 0.0f;
            }
            KQ_max = KQ_max_new;
        }

        float2 KQ_rowsum_add = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int k0 = 0; k0 < KQ_stride; k0 += mma_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < mma_C_KQ::ne; ++l) {
                const float KQ_max_l = l % 2 == 0 ? KQ_max.x : KQ_max.y;
                const float diff = KQ_C[k0/mma_C_KQ::I].x[l] - KQ_max_l;
                KQ_C[k0/mma_C_KQ::I].x[l] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_C[k0/mma_C_KQ::I].x[l] = 0.0f;
                }

                if (l % 2 == 0) {
                    KQ_rowsum_add.x += KQ_C[k0/mma_C_KQ::I].x[l];
                } else {
                    KQ_rowsum_add.y += KQ_C[k0/mma_C_KQ::I].x[l];
                }
            }
        }

        // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
        KQ_rowsum.x = KQ_max_scale.x*KQ_rowsum.x + KQ_rowsum_add.x;
        KQ_rowsum.y = KQ_max_scale.y*KQ_rowsum.y + KQ_rowsum_add.y;

        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale.x, KQ_max_scale.y);
#pragma unroll
        for (int i = 0; i < D/mma_C_VKQ::I; ++i) {
#pragma unroll
            for (int l = 0; l < mma_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }

        // Convert KQ C tiles into B tiles for VKQ calculation:
        mma_B B[KQ_stride/(2*mma_B::K)];
#pragma unroll
        for (int k = 0; k < KQ_stride/(2*mma_B::K); ++k) {
            B[k] = KQ_C[k].to_mma_B();
        }

        // Load V data into tile with decreasing granularity for D for better memory bandwidth:
#pragma unroll
        for (int stride_i : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int i0_start = stride_i == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_i);
            const int i0_stop  =                             D/2 - (D/2) % (1*stride_i);
            const int stride_k = WARP_SIZE/stride_i;

#pragma unroll
            for (int k_V_0 = 0; k_V_0 < KQ_stride; k_V_0 += nwarps*stride_k) {
                const int k_V = k_V_0 + threadIdx.y*stride_k + (stride_i == WARP_SIZE ? 0 : threadIdx.x / stride_i);

#pragma unroll
                for (int i_V_0 = i0_start; i_V_0 < i0_stop; i_V_0 += stride_i) {
                    const int i_V = i_V_0 + (stride_i == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_i);

                    tile_KV[k_V*D2_padded + i_V] = V_h2[(k_VKQ_0 + k_V)*stride_KV + i_V];
                }
            }
        }

        __syncthreads();

        // Calculate VKQ tile:
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += mma_C_VKQ::I) {
#pragma unroll
            for (int k0 = 0; k0 < KQ_stride/2; k0 += mma_A::K) {
                mma_A A;
                A.load_ldmatrix_trans(tile_KV + 2*k0*D2_padded + i_VKQ_0/2, D2_padded);
                VKQ_C[i_VKQ_0/mma_C_VKQ::I].mma(A, B[k0/mma_A::K]);
            }
        }

        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8 threads each, does not need full reduce.
    for (int offset = 16; offset > 2; offset >>= 1) {
        KQ_rowsum.x += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.x, offset, WARP_SIZE);
        KQ_rowsum.y += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.y, offset, WARP_SIZE);
    }

    // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
    // The values after that are for the partial results of the individual blocks.

    const int j_fixup = threadIdx.y*(2*mma_C_VKQ::J) + 2*mma_C_VKQ::get_j(-1);
    const int j_VKQ_0 = jt*ncols + j_fixup;

    if (j_VKQ_0 + 0 >= ne01) {
        return;
    }

    if constexpr (is_fixup) {
        static_assert(!needs_fixup, "needs_fixup and is_fixup are mutually exclusive");
        if (threadIdx.x < mma_C_VKQ::J) {
            dstk_fixup[(gridDim.x + blockIdx.x)*ncols + j_fixup + 0] = make_float2(KQ_max.x, KQ_rowsum.x);
        }
    } else if constexpr (needs_fixup) {
        if (threadIdx.x < mma_C_VKQ::J) {
            dstk_fixup[blockIdx.x*ncols + j_fixup + 0] = make_float2(KQ_max.x, KQ_rowsum.x);
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < D; i0 += mma_C_VKQ::I) {
#pragma unroll
        for (int l = 0; l < mma_C_VKQ::ne; ++l) {
            const int i = i0 + mma_C_VKQ::get_i(l);

            float dst_val = __low2float(VKQ_C[i0/mma_C_VKQ::I].x[l]);
            if constexpr (is_fixup) {
                static_assert(!needs_fixup, "needs_fixup and is_fixup are mutually exclusive");
                float * dstk_fixup_data = ((float *) dstk_fixup) + 2*2*gridDim.x*ncols + blockIdx.x*ncols*D;
                dstk_fixup_data[(j_fixup + 0)*D + i] = dst_val;
            } else {
                if constexpr (!needs_fixup) {
                    dst_val /= KQ_rowsum.x;
                }
                dstk[(j_VKQ_0 + 0)*ne02*D + i] = dst_val;
            }
        }
    }

    if (j_VKQ_0 + 1 >= ne01) {
        return;
    }

    if constexpr (is_fixup) {
        static_assert(!needs_fixup, "needs_fixup and is_fixup are mutually exclusive");
        if (threadIdx.x < mma_C_VKQ::J) {
            dstk_fixup[(gridDim.x + blockIdx.x)*ncols + j_fixup + 1] = make_float2(KQ_max.y, KQ_rowsum.y);
        }
    } else if constexpr (needs_fixup) {
        if (threadIdx.x < mma_C_VKQ::J) {
            dstk_fixup[blockIdx.x*ncols + j_fixup + 1] = make_float2(KQ_max.y, KQ_rowsum.y);
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < D; i0 += mma_C_VKQ::I) {
#pragma unroll
        for (int l = 0; l < mma_C_VKQ::ne; ++l) {
            const int i = i0 + mma_C_VKQ::get_i(l);

            float dst_val = __high2float(VKQ_C[i0/mma_C_VKQ::I].x[l]);
            if constexpr (is_fixup) {
                static_assert(!needs_fixup, "needs_fixup and is_fixup are mutually exclusive");
                float * dstk_fixup_data = ((float *) dstk_fixup) + 2*2*gridDim.x*ncols + blockIdx.x*ncols*D;
                dstk_fixup_data[(j_fixup + 1)*D + i] = dst_val;
            } else {
                if constexpr (!needs_fixup) {
                    dst_val /= KQ_rowsum.y;
                }
                dstk[(j_VKQ_0 + 1)*ne02*D + i] = dst_val;
            }
        }
    }
#else
   NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, ncols <= 32 ? 4 : 2)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_stride == 0, "bad KQ_stride");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int iter_k = ne11 / KQ_stride;
    const int iter_j = (ne01 + (ncols - 1)) / ncols;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*iter_k*iter_j*ne02 / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*iter_k*iter_j*ne02 / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k*iter_j);
        const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

        const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
        const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
        const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
        const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
        float        * dstk  = dst + channel*D;

        const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
                jt, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
                jt, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k*iter_j);
    const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

    const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
    const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
    const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
    float        * dstk  = dst + channel*D;

    const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
        ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne31, nb31, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne0, ne1, ne2, ne3,
        jt, kb0_start, kb0_stop);
}

template <int D, int cols_per_block>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    typedef mma_A_I16K8<half2> mma_A;
    typedef mma_B_J8K8<half2>  mma_B;

    const ggml_tensor * KQV = dst;

    constexpr int    nwarps        = cols_per_block / mma_B::J;
    constexpr int    KQ_stride     = D <= 128 ? 64 : 32;
    constexpr size_t nbytes_shared = std::max(KQ_stride, cols_per_block) * (D + 8) * sizeof(half);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    }
    launch_fattn<D, cols_per_block, 0, KQ_stride>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
}

#define DECL_FATTN_MMA_F16_CASE(D, cols_per_block)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                     \
    <D, cols_per_block>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_MMA_F16_CASE( 64,  8);
extern DECL_FATTN_MMA_F16_CASE( 80,  8);
extern DECL_FATTN_MMA_F16_CASE( 96,  8);
extern DECL_FATTN_MMA_F16_CASE(112,  8);
extern DECL_FATTN_MMA_F16_CASE(128,  8);
extern DECL_FATTN_MMA_F16_CASE(256,  8);

extern DECL_FATTN_MMA_F16_CASE( 64, 16);
extern DECL_FATTN_MMA_F16_CASE( 80, 16);
extern DECL_FATTN_MMA_F16_CASE( 96, 16);
extern DECL_FATTN_MMA_F16_CASE(112, 16);
extern DECL_FATTN_MMA_F16_CASE(128, 16);
extern DECL_FATTN_MMA_F16_CASE(256, 16);

extern DECL_FATTN_MMA_F16_CASE( 64, 32);
extern DECL_FATTN_MMA_F16_CASE( 80, 32);
extern DECL_FATTN_MMA_F16_CASE( 96, 32);
extern DECL_FATTN_MMA_F16_CASE(112, 32);
extern DECL_FATTN_MMA_F16_CASE(128, 32);
extern DECL_FATTN_MMA_F16_CASE(256, 32);

extern DECL_FATTN_MMA_F16_CASE( 64, 64);
extern DECL_FATTN_MMA_F16_CASE( 80, 64);
extern DECL_FATTN_MMA_F16_CASE( 96, 64);
extern DECL_FATTN_MMA_F16_CASE(112, 64);
extern DECL_FATTN_MMA_F16_CASE(128, 64);
extern DECL_FATTN_MMA_F16_CASE(256, 64);
