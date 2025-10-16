#include "llama.h"

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_device_memory_data {
    size_t total;
    size_t free;
    llama_memory_breakdown_data mb;
};

static std::vector<llama_device_memory_data> llama_get_device_memory_data(
        const char * path_model, const llama_model_params * mparams, const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs, uint32_t & hp_ngl, uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level level;
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        if (level >= ud->level) {
            ud->original_logger.callback(level, text, ud->original_logger.user_data);
        }
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc = true;
    mparams_copy.use_mmap = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    llama_memory_breakdown_print(ctx);
    if (ctx == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama_context from model");
    }

    std::vector<llama_device_memory_data> ret(model->devices.size());

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;

        if (ggml_backend_buft_is_host(buft)) {
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < ret.size(); i++) {
            if (model->devices[i] == dev) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }
    for (size_t i = 0; i < ret.size(); i++) {
        ggml_backend_dev_memory(model->devices[i], &ret[i].free, &ret[i].total);
    }

    devs           = model->devices;
    hp_ngl         = model->hparams.n_layer;
    hp_n_ctx_train = model->hparams.n_ctx_train;
    hp_n_expert    = model->hparams.n_expert;

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return ret;
}

bool llama_params_fit_to_free_memory(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overides, enum ggml_log_level log_level) {
    constexpr int64_t MiB = 1024*1024;
    constexpr int64_t target_margin = 1024 * MiB;
    constexpr uint32_t n_ctx_min = 4096;

    typedef std::vector<llama_device_memory_data> dmds_t;

    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl         = 0;
    uint32_t hp_n_ctx_train = 0;
    uint32_t hp_n_expert    = 0;

    const dmds_t dmds_full = llama_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
    const size_t nd = devs.size();
    if (nd == 0) {
        return true; // TODO print
    }

    int64_t sum_projected_free = 0;
    int64_t min_projected_free = INT64_MAX;
    int64_t sum_projected_used = 0;
    int64_t sum_projected_ctx  = 0;

    if (nd > 1) {
        LLAMA_LOG_INFO("%s: projected memory use with initial parameters [MiB]:\n", __func__);
    }
    for (size_t id = 0; id < nd; id++) {
        const llama_device_memory_data & dmd = dmds_full[id];

        const int64_t projected_used = dmd.mb.model + dmd.mb.context + dmd.mb.compute;
        const int64_t projected_free = int64_t(dmd.free) - projected_used;

        sum_projected_used += projected_used;
        sum_projected_free += projected_free;
        min_projected_free  = std::min(min_projected_free, projected_free);
        sum_projected_ctx  += dmd.mb.context;

        if (nd > 1) {
            LLAMA_LOG_INFO("%s:   - %s (%s): total=%zu used=%" PRId64 " free=%" PRId64 "\n",
                __func__, ggml_backend_dev_name(devs[id]), ggml_backend_dev_description(devs[id]),
                dmd.total/MiB, projected_used/MiB, projected_free/MiB);
        }
    }
    if (min_projected_free >= target_margin) {
        if (nd == 1) {
            LLAMA_LOG_INFO("%s: allocation projected to use a total of %" PRId64 " MiB, "
                "will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                __func__, sum_projected_used/MiB, min_projected_free/MiB, target_margin/MiB);
            return true;
        }
        LLAMA_LOG_INFO("%s: allocation projected to use a total of %" PRId64 " MiB, "
            "will leave at least %" PRId64 " >= %" PRId64 " MiB of free memory on all devices, no changes needed\n",
            __func__, sum_projected_used/MiB, min_projected_free/MiB, target_margin/MiB);
        return true;
    }
    {
        int64_t global_surplus = sum_projected_free - int64_t(nd)*target_margin;
        if (global_surplus < 0) {
            if (nd == 1) {
                LLAMA_LOG_INFO("%s: allocation projected to use too much device memory to fulfill margin of %" PRId64 " MiB, "
                    "need to reduce memory use by %" PRId64 " MiB\n",
                    __func__, target_margin/MiB, -global_surplus/MiB);
            } else {
                LLAMA_LOG_INFO("%s: allocation projected to use too much memory to fulfill margin of %" PRId64 " MiB on all devices, "
                    "need to reduce memory use by %" PRId64 " MiB\n",
                    __func__, target_margin/MiB, -global_surplus/MiB);
            }

            if (cparams->n_ctx == 0) {
                if (hp_n_ctx_train > n_ctx_min) {
                    const int64_t bytes_per_ctx = sum_projected_ctx / hp_n_ctx_train;
                    const uint32_t ctx_reduction = std::min(
                        uint32_t((-global_surplus + bytes_per_ctx - 1) / bytes_per_ctx), uint32_t(hp_n_ctx_train - n_ctx_min));
                    cparams->n_ctx = hp_n_ctx_train - ctx_reduction;
                    const int64_t memory_reduction = int64_t(ctx_reduction) * bytes_per_ctx;
                    global_surplus += memory_reduction;
                    LLAMA_LOG_INFO("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %zu MiB less memory\n",
                        __func__, hp_n_ctx_train, cparams->n_ctx, memory_reduction/MiB);
                } else {
                    LLAMA_LOG_INFO("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                        __func__, hp_n_ctx_train, n_ctx_min);
                }
            } else {
                LLAMA_LOG_INFO("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
        if (global_surplus > 0) {
            LLAMA_LOG_INFO("%s: entire model can be fit across devices by reducing context\n", __func__);
            return true;
        }
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        LLAMA_LOG_INFO("%s: n_gpu_layers already set by user to %" PRId32 ", abort\n", __func__, mparams->n_gpu_layers);
        return false;
    }
    if (nd > 1) {
        if (!tensor_split) {
            LLAMA_LOG_INFO("%s: did not provide buffer to automatically set tensor_split, abort\n", __func__);
            return false;
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    LLAMA_LOG_INFO("%s: model_params::tensor_split already set by user, abort\n", __func__);
                    return false;
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            LLAMA_LOG_INFO("%s: changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort\n", __func__);
            return false;
        }
    }
    if (hp_n_expert > 0 && !tensor_buft_overides) {
        LLAMA_LOG_INFO("%s: did not provide buffer to set tensor_buft_overrides for MoE model, abort\n", __func__);
        return false;
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        LLAMA_LOG_INFO("%s: model_params::tensor_buft_overrides already set by user, abort\n", __func__);
        return false;
    }

    auto get_memory_for_const_layer = [&](const int layers_per_device) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        mparams_copy.n_gpu_layers = nd * layers_per_device;
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                tensor_split[id] = 1.0f;
            }
        }
        mparams_copy.tensor_split = tensor_split;
        const dmds_t dmd_nl = llama_get_device_memory_data(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (const llama_device_memory_data & dmd : dmd_nl) {
            ret.push_back(dmd.mb.model + dmd.mb.context + dmd.mb.compute);
        }
        return ret;
    };

    struct memory_scaling {
        int64_t base      = 0;
        int64_t per_layer = 0;
    };
    auto get_memory_scaling = [&](const std::vector<int64_t> & mem_1l, const std::vector<int64_t> & mem_nl, const uint32_t n) -> std::vector<memory_scaling> {
        std::vector<memory_scaling> ret(nd);
        for (size_t id = 0; id < nd; id++) {
            ret[id].per_layer = (mem_nl[id] - mem_1l[id]) / (n - 1);
            ret[id].base      =  mem_1l[id] - ret[id].per_layer;
        }
        return ret;
    };

    if (hp_n_expert > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate)_(ch|)exps";
        tensor_buft_overides[0] = {pattern_moe_all.c_str(), ggml_backend_cpu_buffer_type()};
        tensor_buft_overides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overides;

        const dmds_t dmds_cpu_moe = llama_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
        int64_t global_surplus = 0;
        for (const llama_device_memory_data & dmd : dmds_cpu_moe) {
            global_surplus += int64_t(dmd.free);
            global_surplus -= int64_t(dmd.mb.model + dmd.mb.context + dmd.mb.compute) + target_margin;
        }
        if (global_surplus > 0) {
            LLAMA_LOG_INFO("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n", __func__, global_surplus/MiB);
            const uint32_t nl_scaling = hp_ngl / nd;
            const std::vector<memory_scaling> spl_part = get_memory_scaling(
                get_memory_for_const_layer(1), get_memory_for_const_layer(nl_scaling), nl_scaling);

            tensor_buft_overides[0] = {nullptr, nullptr};
            mparams->tensor_buft_overrides = tensor_buft_overides;

            const std::vector<memory_scaling> spl_full = get_memory_scaling(
                get_memory_for_const_layer(1), get_memory_for_const_layer(nl_scaling), nl_scaling);

            for (size_t id = 0; id < nd - 1; id++) {
                tensor_split[id] = 0.0f;
            }
            tensor_split[nd - 1] = 1.0f;
            const dmds_t dmds_last = llama_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
            tensor_split[nd - 1] = 0.0f;

            struct ngl {
                uint32_t part = 0;
                uint32_t full = 0;
            };
            auto distribute_layers = [&](std::vector<ngl> & ngl_per_device, const uint32_t global_ngl_part) -> bool {
                ngl_per_device.clear();
                ngl_per_device.resize(nd);
                ngl_per_device.back().part = 0;
                ngl_per_device.back().full = hp_ngl + 1;

                std::vector<int64_t> usable_memory;
                usable_memory.reserve(nd);
                for (size_t id = 0; id < nd - 1; id++) {
                    int64_t um = int64_t(dmds_last[id].free) - target_margin - int64_t(spl_full[id].base);
                    um = std::max(um, int64_t(0));
                    usable_memory.push_back(um);
                }
                {
                    const llama_memory_breakdown_data & mb = dmds_last.back().mb;
                    usable_memory.push_back(int64_t(dmds_last.back().free) - int64_t(mb.model + mb.context + mb.context) - target_margin);
                }

                ngl_per_device.back().full -= global_ngl_part;
                usable_memory.back() += spl_full.back().per_layer*global_ngl_part;
                ngl_per_device.back().part += global_ngl_part;
                usable_memory.back() -= spl_part.back().per_layer*global_ngl_part;

                for (size_t id = nd - 1; id >= 1; id--) {
                    while (usable_memory[id] < 0 && ngl_per_device[id].part > 0) {
                        ngl_per_device[id  ].part--;
                        ngl_per_device[id-1].part++;
                        usable_memory[id  ] += spl_part[id  ].per_layer;
                        usable_memory[id-1] -= spl_part[id-1].per_layer;
                    }
                    const uint32_t min_full_layers = id == nd - 1 ? 1 : 0;
                    while (usable_memory[id] < 0 && ngl_per_device[id].full > min_full_layers) {
                        ngl_per_device[id  ].full--;
                        ngl_per_device[id-1].full++;
                        usable_memory[id  ] += spl_full[id  ].per_layer;
                        usable_memory[id-1] -= spl_full[id-1].per_layer;
                    }
                }

                return usable_memory[0] > 0;
            };

            std::vector<ngl> ngl_per_device;
            for (uint32_t global_ngl_part = 0; global_ngl_part < hp_ngl; global_ngl_part++) {
                const bool model_fits = distribute_layers(ngl_per_device, global_ngl_part);
                if (model_fits) {
                    break;
                }
            }

            auto get_moe_pattern = [&](const size_t il) -> const char * {
                static std::vector<std::string> patterns;
                while (patterns.size() <= il) {
                    patterns.push_back("blk\\." + std::to_string(patterns.size()) + "\\.ffn_(up|down|gate)_(ch|)exps");
                }
                return patterns[il].c_str();
            };

            uint32_t il0             = 0;
            uint32_t global_ngl_part = 0;
            uint32_t global_ngl_full = 0;
            llama_model_tensor_buft_override * tbo_cur = tensor_buft_overides;
            for (size_t id = 0; id < nd; id++) {
                for (uint32_t il = il0; il < il0 + ngl_per_device[id].part; il++) {
                    tbo_cur->pattern = get_moe_pattern(il);
                    tbo_cur->buft    = ggml_backend_cpu_buffer_type();
                    tbo_cur++;
                }
                const uint32_t ngl = ngl_per_device[id].part + ngl_per_device[id].full;
                tensor_split[id] = ngl;
                il0 += ngl;

                global_ngl_part += ngl_per_device[id].part;
                global_ngl_full += ngl_per_device[id].full;
            }
            tbo_cur->pattern = nullptr;
            tbo_cur->buft    = nullptr;
            mparams->tensor_buft_overrides = tensor_buft_overides;

            const llama_memory_breakdown_data & mb_last = dmds_last.back().mb;
            const int64_t projected_use_last = int64_t(mb_last.model + mb_last.context + mb_last.compute)
                - int64_t(hp_ngl + 1 - ngl_per_device.back().full) * spl_full.back().per_layer
                - int64_t(ngl_per_device.back().part) * (spl_full.back().per_layer - spl_part.back().per_layer);
            const int64_t projected_margin_last = int64_t(dmds_last.back().free) - projected_use_last;

            if (nd == 1) {
                LLAMA_LOG_INFO("%s: set to use %u dense-only layers and %u full layers, %" PRId64 " MiB used, %" PRId64 " MiB free\n",
                    __func__, ngl_per_device.back().part, ngl_per_device.back().full, projected_use_last/MiB, projected_margin_last/MiB);
                return true;
            }
            LLAMA_LOG_INFO("%s: set to use %u dense-only and %u full GPU layers in total, projected memory use:\n",
                __func__, global_ngl_part, global_ngl_full);
            for (size_t id = 0; id < nd - 1; id++) {
                const int64_t projected_use = spl_full[id].base
                    + int64_t(ngl_per_device[id].part)*spl_part[id].per_layer + ngl_per_device[id].full*spl_full[id].per_layer;
                const int64_t projected_margin = int64_t(dmds_last[id].free) - projected_use;
                LLAMA_LOG_INFO("%s:   - %s (%s): %u dense-only layers, %u full layers, %zu MiB used, %zu MiB free\n",
                    __func__, ggml_backend_dev_name(devs[id]), ggml_backend_dev_description(devs[id]),
                    ngl_per_device[id].part, ngl_per_device[id].full, projected_use/MiB, projected_margin/MiB);
            }
            LLAMA_LOG_INFO("%s:   - %s (%s): %u dense-only layers, %u full layers, %zu MiB used, %zu MiB free\n",
                __func__, ggml_backend_dev_name(devs.back()), ggml_backend_dev_description(devs.back()),
                ngl_per_device.back().part, ngl_per_device.back().full, projected_use_last/MiB, projected_margin_last/MiB);
            return true;
        }
        LLAMA_LOG_INFO("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n", __func__, -global_surplus/MiB);
    }

    const uint32_t nl_scaling = hp_ngl / nd;
    const std::vector<memory_scaling> ms = get_memory_scaling(
        get_memory_for_const_layer(1), get_memory_for_const_layer(nl_scaling), nl_scaling);

    mparams->n_gpu_layers = 0;
    std::vector<uint32_t> ngl_per_device;
    ngl_per_device.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        const uint32_t ngl = (dmds_full[id].free - target_margin - ms[id].base) / ms[id].per_layer;
        mparams->n_gpu_layers += ngl;
        ngl_per_device.push_back(ngl);
    }
    if (nd == 1) {
        const int64_t projected_use = ms[0].base + int64_t(ngl_per_device[0])*ms[0].per_layer;
        const int64_t projected_margin = int64_t(dmds_full[0].free) - projected_use;
        LLAMA_LOG_INFO("%s: set n_gpu_layers to %" PRId32 ", projected to use %" PRId64 " MiB with %" PRId64 " MiB free\n",
            __func__, mparams->n_gpu_layers, projected_use/MiB, projected_margin/MiB);
        return true;
    }
    LLAMA_LOG_INFO("%s: set n_gpu_layers to %" PRId32 ", projected memory use:\n", __func__, mparams->n_gpu_layers);
    for (size_t id = 0; id < nd; id++) {
        const int64_t projected_use = ms[id].base + int64_t(ngl_per_device[id])*ms[id].per_layer;
        const int64_t projected_margin = int64_t(dmds_full[id].free) - projected_use;
        LLAMA_LOG_INFO("%s:   - %s (%s): %d layers, %zu MiB used, %zu MiB free\n",
            __func__, ggml_backend_dev_name(devs[id]), ggml_backend_dev_description(devs[id]),
            ngl_per_device[id], projected_use/MiB, projected_margin/MiB);
    }
    return true;
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;
        model.hparams.no_alloc   = params.no_alloc;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        std::vector<ggml_backend_dev_t> rpc_servers;

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU: {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        // check if there is already a GPU with the same device id
                        ggml_backend_dev_props props;
                        ggml_backend_dev_get_props(dev, &props);
                        auto it = std::find_if(gpus.begin(), gpus.end(), [&props](ggml_backend_dev_t d) {
                            ggml_backend_dev_props d_props;
                            ggml_backend_dev_get_props(d, &d_props);
                            if (props.device_id && d_props.device_id) {
                                return strcmp(props.device_id, d_props.device_id) == 0;
                            }
                            return false;
                        });

                        if (it != gpus.end()) {
                            LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                    __func__,
                                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                    props.device_id ? props.device_id : "unknown id",
                                    ggml_backend_dev_name(*it), ggml_backend_dev_description(*it));
                        } else {
                            gpus.push_back(dev);
                        }
                    }
                    break;
                }

                case GGML_BACKEND_DEVICE_TYPE_IGPU:
                    igpus.push_back(dev);
                    break;
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(*model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

