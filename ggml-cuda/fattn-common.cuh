#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

typedef void (* fattn_kernel_t)(
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
        const int ne3);

typedef half (*vec_dot_fattn_vec_KQ_t)(
    const char * __restrict__ K_c, const half2 * __restrict__ Q_h2, const int * __restrict__ Q_q8 , const half2 * __restrict__ Q_ds);

template<int D>
static __device__ __forceinline__ half vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const half2 * __restrict__ Q_h2, const int * __restrict__ Q_q8, const half2 * __restrict__ Q_ds) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_h2);

    half sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ / QI8_1;
        const int iqs4  = k_KQ % QI4_0;
        const int shift = k_KQ & QI4_0;

        const int v = (get_int_from_uint8(K_q4_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int   sumi = __dp4a(v, u, 0);
        const half2 sum2 = __half2half2(K_q4_0[ib].d) * Q_ds[k_KQ_0/WARP_SIZE];
        sum += ((half) sumi)*__low2half(sum2) - __high2half(sum2) /* *8/QI8_1 == 1 */;
    }

    return sum;
}

template<int D>
static __device__ __forceinline__ half vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const half2 * __restrict__ Q_h2, const int * __restrict__ Q_q8, const half2 * __restrict__ Q_ds) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_h2);

    half sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ / QI8_1;
        const int iqs4  = k_KQ % QI4_1;
        const int shift = k_KQ & QI4_1;

        const int v = (get_int_from_uint8(K_q4_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int   sumi = __dp4a(v, u, 0);
        const half2 d4d8_m4s8 = K_q4_1[ib].dm * Q_ds[k_KQ_0/WARP_SIZE];
        const half2 sumid4d8_m4s8scaled = d4d8_m4s8 * make_half2(sumi, 1.0f/QI8_1);
        sum += __low2half(sumid4d8_m4s8scaled) + __high2half(sumid4d8_m4s8scaled);
    }

    return sum;
}

template<int D>
static __device__ __forceinline__ half vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const half2 * __restrict__ Q_h2, const int * __restrict__ Q_q8, const half2 * __restrict__ Q_ds) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_h2);

    half sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        const int v = get_int_from_int8(K_q8_0[ib].qs, iqs);

        sum += vec_dot_q8_0_q8_1_impl<half, 1>(&v, &Q_q8[k_KQ_0/WARP_SIZE], K_q8_0[ib].d, __low2half(Q_ds[k_KQ_0/WARP_SIZE]));
    }

    return sum;
}

template<int D>
static __device__ __forceinline__ half vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const half2 * __restrict__ Q_h2, const int * __restrict__ Q_q8 , const half2 * __restrict__ Q_ds) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds);

    half2 sum2 = make_half2(0.0f, 0.0f);

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const half2 K_ik = K_h2[k_KQ];
        sum2 += K_ik * Q_h2[k_KQ_0/WARP_SIZE];
    }

    return __low2half(sum2) + __high2half(sum2);
}

typedef half  (*dequantize_1_f16_t)(const void *, const int64_t);
typedef float (*dequantize_1_f32_t)(const void *, const int64_t);

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_0(const void * __restrict__ vx, const int64_t i) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i          /  QK4_0;
    const int     iqs   =  i          % (QK4_0/2);
    const int     shift = (i % QK4_0) / (QK4_0/2);

    const T   d  = x[ib].d;
    const int q0 = x[ib].qs[iqs];
    const int q  = ((q0 >> 4*shift) & 0x0F) - 8;

    return d*((T) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q8_0(const void * __restrict__ vx, const int64_t i) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i / QK8_0;
    const int     iqs = i % QK8_0;

    const T   d = x[ib].d;
    const int q = x[ib].qs[iqs];

    return d*((T) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_f16(const void * __restrict__ vx, const int64_t i) {
    const half * x = (const half *) vx;

    return x[i];
}

template<int D, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
    VKQ_parts += parallel_blocks*D * gridDim.y*blockIdx.x;
    VKQ_meta  += parallel_blocks   * gridDim.y*blockIdx.x;
    dst       +=                 D * gridDim.y*blockIdx.x;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float2 meta[parallel_blocks];
    if (tid < 2*parallel_blocks) {
        ((float *) meta)[threadIdx.x] = ((const float *)VKQ_meta) [blockIdx.y*(2*parallel_blocks) + tid];
    }

    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
#pragma unroll
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        const float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*D + blockIdx.y*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[blockIdx.y*D + tid] = VKQ_numerator / VKQ_denominator;
}

template <int D, int parallel_blocks>
void launch_fattn(ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, int nwarps, int cols_per_block) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const dim3 blocks_num(parallel_blocks*((Q->ne[1] + cols_per_block - 1) / cols_per_block), Q->ne[2], Q->ne[3]);
    const int  shmem = 0;

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) KQV->op_params + 1, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    fattn_kernel<<<blocks_num, block_dim, shmem, main_stream>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? ((const char *) mask->data) : nullptr,
        (parallel_blocks) == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
        mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
        Q->nb[1], Q->nb[2], Q->nb[3],
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
    );
    CUDA_CHECK(cudaGetLastError());

    if ((parallel_blocks) == 1) {
        return;
    }

    const dim3 block_dim_combine(D, 1, 1);
    const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);
    const int  shmem_combine = 0;

    flash_attn_combine_results<D, parallel_blocks>
        <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
        (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
    CUDA_CHECK(cudaGetLastError());
}
