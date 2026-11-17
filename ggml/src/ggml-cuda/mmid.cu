#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template, bool low_sram>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;

    const int expert    = blockIdx.x; // The expert this CUDA block handles.
    const int expert_nr = blockIdx.y; // If low_sram, then each CUDA block only handles a single number of previous repetitions of the expert.

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int nr_prev    = 0; // Number of columns with the right expert but lower repetitions (only used for low_sram)
    int it_compact = 0; // Running index for the compact slice of this expert (with expert_nr previous repetitions if low_sram).

    int it_prev = -1;
    int n_found_token = 0;

    auto check_experts = [&] (const int it, const int iex, const int expert_found) {
        nex_prev += expert_found < expert;

        const ggml_cuda_thread_mask_t iex_used = ggml_cuda_ballot(expert_found == expert);
        const ggml_cuda_thread_mask_t mask = (ggml_cuda_thread_mask_t(1) << threadIdx.x) - 1;

        if constexpr (low_sram) {
            int rep_idx = 0;
            if constexpr (n_expert_used_template == 0) {
                if (it != it_prev) {
                    it_prev = it;
                    n_found_token = 0;
                }
                rep_idx = n_found_token + ggml_cuda_popc(iex_used & mask);
                n_found_token += ggml_cuda_popc(iex_used);
            } else {
                constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
                const int token_in_warp = threadIdx.x / neu_padded;
                const int token_start_lane = token_in_warp * neu_padded;
                const int lane_in_token = threadIdx.x % neu_padded;
                const ggml_cuda_thread_mask_t token_mask = ((ggml_cuda_thread_mask_t(1) << lane_in_token) - 1) << token_start_lane;
                rep_idx = ggml_cuda_popc(iex_used & token_mask);
            }

            const bool is_expert = expert_found == expert;
            nr_prev += is_expert && (rep_idx < expert_nr);

            const bool write_store = is_expert && (rep_idx == expert_nr);
            const ggml_cuda_thread_mask_t store_mask = ggml_cuda_ballot(write_store);
            const int it_compact_offset = ggml_cuda_popc(store_mask & mask);

            if (write_store) {
                store[it_compact + it_compact_offset] = mm_ids_helper_store(it, iex);
            }

            it_compact += ggml_cuda_popc(store_mask);
        } else {
            GGML_UNUSED(expert_nr);

            const int it_compact_offset = ggml_cuda_popc(iex_used & mask);

            if (expert_found == expert) {
                store[it_compact + it_compact_offset] = mm_ids_helper_store(it, iex);
            }

            it_compact += ggml_cuda_popc(iex_used);
        }
    };

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;
                check_experts(it, iex, expert_found);
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.

        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            check_experts(it, iex, expert_found);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    if constexpr (low_sram) {
        nr_prev        = warp_reduce_sum<warp_size>(nr_prev);
    }
    ggml_cuda_syncwarp();

    const int n_prev = low_sram ? (nex_prev + nr_prev) : nex_prev;

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[n_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = n_prev + itc;
        } else {
            ids_src1[n_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if constexpr (low_sram) {
        if (expert_nr != gridDim.y - 1) {
            return;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    const int n_expert_cols = nr_prev + it_compact;

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), n_expert_cols);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + n_expert_cols);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared_high = n_tokens*n_expert_used_var*sizeof(mm_ids_helper_store);
    if (nbytes_shared_high <= smpbo) {
        constexpr bool low_sram = false;
        const dim3 num_blocks(n_experts, 1, 1);
        CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, low_sram>), smpbo);
        mm_ids_helper<n_expert_used_template, low_sram><<<num_blocks, block_size, nbytes_shared_high, stream>>>
            (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
        return;
    }
    const size_t nbytes_shared_low = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared_low <= smpbo);

    constexpr bool low_sram = true;
    const dim3 num_blocks(n_experts, n_expert_used_var, 1);
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, low_sram>), smpbo);
    mm_ids_helper<n_expert_used_template, low_sram><<<num_blocks, block_size, nbytes_shared_low, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}
