#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t> simple_devs;

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return simple_devs < other.simple_devs;
    }
};

//
// meta backend device
//

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    return "Meta";

    GGML_UNUSED(dev);
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    const size_t n_devs = ggml_backend_meta_device_n_devs(dev);
    std::vector<ggml_backend_t> simple_backends;
    simple_backends.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_backends.push_back(ggml_backend_dev_init(ggml_backend_meta_device_simple_dev(dev, i), params));
    }
    return ggml_backend_meta_init(simple_backends.data(), simple_backends.size());
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_device_is_meta(dev));
    const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(dev_ctx->simple_devs.begin(), dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_device_is_meta(dev));
    const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(dev_ctx->simple_devs.begin(), dev_ctx->simple_devs.end(),
        [buft](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_buft(simple_dev, buft); });
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ nullptr,
    /* .get_memory           = */ nullptr,
    /* .get_type             = */ nullptr,
    /* .get_props            = */ nullptr,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_device_is_meta(ggml_backend_dev_t dev) {
    return dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_device_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_device_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_device_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_device_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t * devs, size_t n_devs) {
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    ggml_backend_meta_device_context ctx;
    ctx.simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        ctx.simple_devs.push_back(devs[i]);
    }
    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Meta";

    GGML_UNUSED(buft);
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ nullptr,
    /* .get_alignment    = */ nullptr,
    /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
    /* .is_host          = */ nullptr,
};

bool ggml_backend_buffer_type_is_meta(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_device_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }
    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ nullptr,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}


struct ggml_backend_meta_buffer_context {
    ggml_context * orig_ctx;
    std::map<const ggml_tensor *, ggml_backend_meta_split_state> split_states;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>>    simple_tensors;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;
        size_t                  tensor_split;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf, size_t tensor_split) : ctx(ctx), buf(buf), tensor_split(tensor_split) {}
    };
    std::vector<buffer_config> buf_configs;

    std::vector<size_t> tensor_split_scan() {
        std::vector<size_t> ret;
        ret.reserve(buf_configs.size() + 1);
        ret.push_back(0);
        for (size_t i = 0; i < buf_configs.size(); i++) {
            ret.push_back(ret.back() + buf_configs[i].tensor_split);
        }
        return ret;
    }
};

// bool ggml_backend_meta_buffer_simple_tensors(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, struct ggml_tensor ** simple_tensors) {
//     ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
//     const size_t n_bufts = ggml_backend_meta_buffer_type_n_bufts(buffer->buft);
//     for (size_t j = 0; j < n_bufts; j++) {
//         simple_tensors[j] = ggml_get_first_tensor(buf_ctx->buf_configs[j].ctx);
//         assert(simple_tensors[j] != nullptr);
//     }
//     for (ggml_tensor * t = ggml_get_first_tensor(buf_ctx->orig_ctx); t != nullptr; t = ggml_get_next_tensor(buf_ctx->orig_ctx, t)) {
//         if (t == tensor) {
//             return true;
//         }
//         for (size_t j = 0; j < n_bufts; j++) {
//             simple_tensors[j] = ggml_get_next_tensor(buf_ctx->buf_configs[j].ctx, simple_tensors[j]);
//             assert(simple_tensors[j] != nullptr);
//         }
//     }
//     return false;
// }

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf, _] : buf_ctx->buf_configs) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) buffer->context;

    ggml_backend_meta_split_state split_state;
    {
        auto it = buf_ctx->split_states.find(tensor);
        GGML_ASSERT(it != buf_ctx->split_states.end());
        split_state = it->second;
    }
    std::vector<ggml_tensor *> simple_tensors;
    {
        auto it = buf_ctx->simple_tensors.find(tensor);
        assert(it != buf_ctx->simple_tensors.end());
        simple_tensors = it->second;
    }

    switch (split_state) {
        case GGML_BACKEND_SPLIT_STATE_MIRRORED: {
            for (ggml_tensor * t : simple_tensors) {
                ggml_backend_tensor_set(t, data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) buffer->context;

    ggml_backend_meta_split_state split_state;
    {
        auto it = buf_ctx->split_states.find(tensor);
        GGML_ASSERT(it != buf_ctx->split_states.end());
        split_state = it->second;
    }
    std::vector<ggml_tensor *> simple_tensors;
    {
        auto it = buf_ctx->simple_tensors.find(tensor);
        assert(it != buf_ctx->simple_tensors.end());
        simple_tensors = it->second;
    }

    switch (split_state) {
        case GGML_BACKEND_SPLIT_STATE_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_tensor_get(simple_tensors[0], data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ nullptr,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ nullptr,
    /* .reset           = */ nullptr,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

ggml_backend_buffer_t ggml_backend_meta_alloc_ctx_tensors_from_buft(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, const size_t * tensor_split, const enum ggml_backend_meta_split_state * split_states) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(buft));
    std::vector<ggml_tensor *> original_tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        GGML_ASSERT(ggml_is_contiguous(t));
        GGML_ASSERT(t->ne[2] == 1);
        GGML_ASSERT(t->ne[3] == 1);
        original_tensors.push_back(t);
    }
    ggml_init_params params = {
        /*.mem_size   =*/ original_tensors.size()*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_dev_t meta_dev = ggml_backend_buft_get_device(buft);
    const size_t n_devs = ggml_backend_meta_device_n_devs(meta_dev);
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context;
    buf_ctx->orig_ctx = ctx;
    for (size_t i = 0; i < n_devs; i++) {
        buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr, tensor_split[i]);
    }
    for (size_t i = 0; i < original_tensors.size(); i++) {
        buf_ctx->split_states[original_tensors[i]] = split_states[i];
    }

    const std::vector<size_t> tensor_split_scan = buf_ctx->tensor_split_scan();

    for (size_t i = 0; i < original_tensors.size(); i++) {
        int split_dim = split_states[i];
        int64_t ne[GGML_MAX_DIMS];
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ne[k] = original_tensors[i]->ne[k];
        }
        std::vector<ggml_tensor *> simple_tensors;
        simple_tensors.reserve(n_devs);
        for (size_t j = 0; j < n_devs; j++) {
            if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                const int64_t low  = ne[split_dim] * tensor_split_scan[j]     / tensor_split_scan.back();
                const int64_t high = ne[split_dim] * tensor_split_scan[j + 1] / tensor_split_scan.back();
                ne[split_dim] = high - low;
            }

            ggml_tensor * t_j = ggml_new_tensor(buf_ctx->buf_configs[j].ctx, original_tensors[i]->type, GGML_MAX_DIMS, ne);
            ggml_set_name(t_j, original_tensors[i]->name);
            simple_tensors.push_back(t_j);
        }
        buf_ctx->simple_tensors[original_tensors[i]] = simple_tensors;
    }

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_meta_device_simple_dev(meta_dev, i));
        buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(buf_ctx->buf_configs[i].ctx, buft);
    }
    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, nullptr, 0);
}

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    std::vector<ggml_backend_t> simple_backends;
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    return "Meta";

    GGML_UNUSED(backend);
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

// static ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const ggml_tensor * t) {
//     auto get_split_state = [&](const size_t i) -> ggml_backend_meta_split_state {
//         if (t->src[i] == nullptr) {
//             return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
//         }
//         return ggml_backend_meta_buffer_type_split_state(ggml_backend_buffer_get_type(t->src[i]->buffer));
//     };
//     switch (t->op) {
//         case GGML_OP_MUL_MAT: {
//             if (get_split_state(0) == GGML_BACKEND_SPLIT_STATE_BY_NE1 && get_split_state(1) == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
//                 return GGML_BACKEND_SPLIT_STATE_BY_NE0;
//             }
//             if (get_split_state(0) == GGML_BACKEND_SPLIT_STATE_BY_NE0 && get_split_state(1) == GGML_BACKEND_SPLIT_STATE_BY_NE0) {
//                 return GGML_BACKEND_SPLIT_STATE_PARTIAL;
//             }
//             if (get_split_state(0) == GGML_BACKEND_SPLIT_STATE_BY_NE2 && get_split_state(1) == GGML_BACKEND_SPLIT_STATE_BY_NE2) {
//                 return GGML_BACKEND_SPLIT_STATE_BY_NE2;
//             }
//         } break;
//         default: {
//         } break;
//     }
//     return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
// }

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(backend));

    int i_start = 0;
    while (i_start < cgraph->size) {

    }
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ggml_backend_t ggml_backend_meta_init(ggml_backend_t * simple_backends, size_t n_backends) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context;
    backend_ctx->simple_backends.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        backend_ctx->simple_backends.push_back(simple_backends[i]);
    }

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = nullptr;
    backend->context = backend_ctx;
    return backend;
}
