#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
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
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = ncols == 1 && type_K == GGML_TYPE_F16 ? 128 / cpy_nb : WARP_SIZE;
    constexpr int nthreads_V  = ncols == 1 && type_V == GGML_TYPE_F16 ? 128 / cpy_nb : WARP_SIZE;

    static_assert(WARP_SIZE % nthreads_V == 0, "bad nthreads_V");
    constexpr int V_cols_per_iter = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<D, nthreads_KQ>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef FAST_FP16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<half,  2*cpy_ne>(type_V);
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<float, 2*cpy_ne>(type_V);
#endif // FAST_FP16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

#ifdef FAST_FP16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[nwarps*V_cols_per_iter*ncols*D];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[nwarps*V_cols_per_iter*ncols*D];
#endif // FAST_FP16_AVAILABLE

    float kqmax[ncols];
    float kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
    }

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef FAST_FP16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // FAST_FP16_AVAILABLE
    int    Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D >= D/(sizeof(int)*QK8_1)];
    float2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            static_assert(nthreads_KQ == WARP_SIZE, "bad nthreads_KQ");
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef FAST_FP16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (threadIdx.x % nthreads_KQ)*cpy_ne;

                float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols <= 2 || ic0 + j < ne01) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols <= 2 || ic0 + j < ne01) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // FAST_FP16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQr[ncols]; // FIXME name

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (threadIdx.x & ~(nthreads_KQ-1)) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x % nthreads_KQ == i_KQ_0) {
                    KQr[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], __shfl_xor_sync(0xFFFFFFFF, kqmax_new_arr[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_arr[j]);
            kqmax[j] = kqmax_new_arr[j];

            KQr[j] = expf(KQr[j] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + KQr[j];
            KQ[j*nthreads + tid] = KQr[j];

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + threadIdx.x / nthreads_V;

#ifdef FAST_FP16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*cpy_ne) {
                half2 tmp[cpy_ne];
                dequantize_V(V + k*nb21, tmp, 2*(i_VKQ_0 + (threadIdx.x % nthreads_V)*cpy_ne));
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < cpy_ne; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*cpy_ne) {
                float2 tmp[cpy_ne];
                dequantize_V(V + k*nb21, tmp, 2*(i_VKQ_0 + (threadIdx.x % nthreads_V)*cpy_ne));
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < cpy_ne; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0 && threadIdx.y < ncols) {
        const float sink = ((const float *) sinks)[head];

        const float kqmax_new_j = fmaxf(sink, kqmax[threadIdx.y]);
        const float KQ_max_scale = expf(kqmax[threadIdx.y] - kqmax_new_j);
        kqmax[threadIdx.y] = kqmax_new_j;

        kqsum[threadIdx.y] = kqsum[threadIdx.y]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - kqmax[threadIdx.y]) : 0.0f);

#ifdef FAST_FP16_AVAILABLE
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[threadIdx.y][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
        }
#else
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[threadIdx.y][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
            VKQ[threadIdx.y][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
        }
#endif // FAST_FP16_AVAILABLE
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            kqmax_shared[j][threadIdx.y] = kqmax[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float kqmax_new = kqmax_shared[j][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(kqmax[j] - kqmax_new);
        kqmax[j] = kqmax_new;

#ifdef FAST_FP16_AVAILABLE
        half2 * KQ_j = (half2 *) KQ + j*(nwarps*V_cols_per_iter*D/2) + threadIdx.y*(V_cols_per_iter*D/2) + (threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*cpy_ne) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*cpy_ne;

            ggml_cuda_memcpy_1<cpy_nb>(KQ_j + i_VKQ, &VKQ[j][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * KQ_j = (float2 *) KQ + j*(nwarps*V_cols_per_iter*D/2) + threadIdx.y*(V_cols_per_iter*D/2) + (threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*cpy_ne) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*cpy_ne;

            ggml_cuda_memcpy_1<cpy_nb>(KQ_j + i_VKQ,            &VKQ[j][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<cpy_nb>(KQ_j + i_VKQ + cpy_ne/2, &VKQ[j][i_VKQ_0/nthreads_V + cpy_ne/2]);
        }
#endif // FAST_FP16_AVAILABLE

        kqsum[j] *= kqmax_scale;
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    if (nthreads > D && tid >= D) {
        return;
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += nthreads) {
            float dst_val = 0;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                for (int v = 0; v < V_cols_per_iter; ++v) {
                    dst_val += float(KQ[j_VKQ*nwarps*V_cols_per_iter*D + w*V_cols_per_iter*D + v*D + i0 + tid]);
                }
            }
            if (gridDim.y == 1) {
                dst_val /= kqsum[j_VKQ];
            }
            dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
        }
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
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
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);
