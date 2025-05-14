#include "common.cuh"
#include "fattn-common.cuh"

template<int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f32(
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
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
        GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
        GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
        GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
        GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02);
        GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne11);
        GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
        GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02);
        GGML_UNUSED(nb03); GGML_UNUSED(nb11); GGML_UNUSED(nb12);
        GGML_UNUSED(nb13); GGML_UNUSED(nb21); GGML_UNUSED(nb22);
        GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
        GGML_UNUSED(ne2); GGML_UNUSED(ne3);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols = ncols1*ncols2;
    constexpr vec_dot_KQ_f32_t vec_dot_KQ = get_vec_dot_KQ_f32<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(type_V);

    const int ic0 = blockIdx.x * ncols1; // Index of the Q/QKV column to work on.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb02*(blockIdx.z *  ncols2)            + nb01*ic0;
    K += nb12*(blockIdx.z * (ncols2/gqa_ratio));
    V += nb22*(blockIdx.z * (ncols2/gqa_ratio)); // K and V have same shape
    const half * maskh = (const half *) mask + ne11*ic0;

    const float slope = get_alibi_slope(max_bias, blockIdx.z, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float KQ[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols] = {0.0f};

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ float maskf_shared[ncols1*D];

    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    float2  Q_f2[ncols][D/(2*WARP_SIZE)];
    int    Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D >= D/(sizeof(int)*QK8_1)];
    float2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps) {
            const int jc = jc0 + threadIdx.y;

            if (jc0 + nwarps > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols1 > 2 && ic0 + j >= ne01) {
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

            const float * Q_f = (const float *) (Q + j*nb01 + c*nb02);
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            int    * tmp_q_i32 = (int    *) &KQ[jc*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[jc][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[jc][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            const int j = jc / ncols2;
            const int c = jc % ncols2;

            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01 + c*nb02);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_f2[jc][i0/WARP_SIZE]    = ncols1 <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_f2[jc][i0/WARP_SIZE].x *= scale;
                Q_f2[jc][i0/WARP_SIZE].y *= scale;
            }
        }
    }

    float VKQ[ncols] = {0.0f};

    for (int k_VKQ_0 = blockIdx.y*D; k_VKQ_0 < ne11; k_VKQ_0 += gridDim.y*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float mask_j = (ncols2 > 1 || mask) ? slope*__half2float(maskh[j*ne11 + k_VKQ_0 + tid]) : 0.0f;
            maskf_shared[j*D + tid] = mask_j;
        }

        __syncthreads();

        bool skip = true;
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
#pragma unroll
            for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                skip = skip && isinf(maskf_shared[j*D + i]);
            }
        }
        if (__all_sync(0xFFFFFFFF, skip)) {
            continue;
        }

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            kqmax_new_arr[jc] = kqmax[jc];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int jc = 0; jc < ncols; ++jc) {
                const int j = jc / ncols2;

                float sum = vec_dot_KQ(K + (k_VKQ_0 + i_KQ)*nb11, Q_f2[jc], Q_i32[jc], Q_ds[jc]);
                sum = warp_reduce_sum(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskf_shared[j*D + i_KQ];

                kqmax_new_arr[jc] = fmaxf(kqmax_new_arr[jc], sum);

                if (threadIdx.x == 0) {
                    KQ[jc*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            float kqmax_new_jc = kqmax_new_arr[jc];

            if (threadIdx.x == 0) {
                kqmax_shared[jc][threadIdx.y] = kqmax_new_jc;
            }
        }

        __syncthreads();

#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            float kqmax_new_jc = kqmax_shared[jc][threadIdx.x];
            kqmax_new_jc = warp_reduce_max(kqmax_new_jc);

            const float KQ_max_scale = expf(kqmax[jc] - kqmax_new_jc);
            kqmax[jc] = kqmax_new_jc;

            const float val = expf(KQ[jc*D + tid] - kqmax[jc]);
            kqsum[jc] = kqsum[jc]*KQ_max_scale + val;
            KQ[jc*D + tid] = val;

            VKQ[jc] *= KQ_max_scale;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < D; ++k) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = dequantize_1_v(V + (k_VKQ_0 + k)*nb21, tid);
#pragma unroll
            for (int jc = 0; jc < ncols; ++jc) {
                VKQ[jc] += V_ki*KQ[jc*D + k];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int jc = 0; jc < ncols; ++jc) {
        kqsum[jc] = warp_reduce_sum(kqsum[jc]);
        if (threadIdx.x == 0) {
            kqsum_shared[jc][threadIdx.y] = kqsum[jc];
        }
    }

    __syncthreads();

#pragma unroll
    for (int jc_VKQ = 0; jc_VKQ < ncols; ++jc_VKQ) {
        const int j_VKQ = jc_VKQ / ncols2;
        const int c_VKQ = jc_VKQ % ncols2;

        if (ncols1 > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[jc_VKQ] = kqsum_shared[jc_VKQ][threadIdx.x];
        kqsum[jc_VKQ] = warp_reduce_sum(kqsum[jc_VKQ]);

        float dst_val = VKQ[jc_VKQ];
        if (gridDim.y == 1) {
            dst_val /= kqsum[jc_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*gridDim.y + blockIdx.y;
        const int c_dst = blockIdx.z*ncols2 + c_VKQ;
        dst[j_dst*gridDim.z*(D*ncols2) + c_dst*D + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols1 <= 2 || ic0 + tid/ncols2 < ne01)) {
        const int j_VKQ = tid / ncols2;
        const int c_VKQ = tid % ncols2;

        dst_meta[(((ic0 + j_VKQ)*gridDim.z + blockIdx.z)*ncols2 + c_VKQ) * gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(ne10);
    GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13); GGML_UNUSED(nb21);
    GGML_UNUSED(nb22); GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_vec_ext_f32<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_vec_ext_f32<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>;
    }

    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 == 1) {
        constexpr int ncols1 = 1;
        if (Q->ne[1] <= ncols1) {
            ggml_cuda_flash_attn_ext_vec_f32_launch<D, ncols1, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 2) {
        constexpr int ncols1 = 2/ncols2;
        if (Q->ne[1] <= ncols1) {
            ggml_cuda_flash_attn_ext_vec_f32_launch<D, ncols1, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 4) {
        constexpr int ncols1 = 4/ncols2;
        if (Q->ne[1] <= ncols1) {
            ggml_cuda_flash_attn_ext_vec_f32_launch<D, ncols1, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    constexpr int ncols1 = 8/ncols2;
    ggml_cuda_flash_attn_ext_vec_f32_launch<D, ncols1, ncols2, type_K, type_V>(ctx, dst);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_vec_f32_switch_ncols1<D, 8, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_vec_f32_switch_ncols1<D, 4, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_vec_f32_switch_ncols1<D, 2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_vec_f32_switch_ncols1<D, 1, type_K, type_V>(ctx, dst);
}

#define DECL_FATTN_VEC_F32_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);
