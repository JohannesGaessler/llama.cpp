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
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

//
// meta backend device
//

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    return "Meta";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    return "Meta";

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 1;
    *total = 1;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_device_is_meta(dev));
    const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(dev_ctx->simple_devs.begin(), dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_device_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_device_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * dev_buft_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (dev_ctx->simple_devs.size() != dev_buft_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < dev_ctx->simple_devs.size(); i++) {
        if (dev_ctx->simple_devs[i] != dev_buft_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
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
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
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

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    ggml_backend_meta_device_context ctx;
    ctx.simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        ctx.simple_devs.push_back(devs[i]);
    }
    ctx.get_split_state    = get_split_state;
    ctx.get_split_state_ud = get_split_state_ud;

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

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Meta";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buffer_type_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buffer_type_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buffer_type_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buffer_type_simple_buft(buft, i)));
    }
    return max_size;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
    /* .is_host          = */ nullptr,
};

bool ggml_backend_buffer_type_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
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

    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context;
    const size_t n_devs = ggml_backend_meta_device_n_devs(dev);
    buft_ctx->simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        buft_ctx->simple_bufts.push_back(
            ggml_backend_dev_buffer_type(ggml_backend_meta_device_simple_dev(dev, i)));
    }

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

size_t ggml_backend_meta_buffer_type_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

ggml_backend_buffer_type_t ggml_backend_meta_buffer_type_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

struct ggml_backend_meta_buffer_context {
    std::map<const ggml_tensor *, ggml_backend_meta_split_state> split_state_cache;
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

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_fix_via_sync =*/ true);
    GGML_ASSERT(split_state != GGML_BACKEND_SPLIT_STATE_UNKNOWN && split_state != GGML_BACKEND_SPLIT_STATE_UNKNOWN);

    int split_dim = split_state;
    int64_t ne[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
    }

    const std::vector<size_t> tensor_split_scan = buf_ctx->tensor_split_scan();

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(buf_ctx->buf_configs.size());
    for (size_t j = 0; j < buf_ctx->buf_configs.size(); j++) {
        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            const int64_t low  = tensor->ne[split_dim] * tensor_split_scan[j]     / tensor_split_scan.back();
            const int64_t high = tensor->ne[split_dim] * tensor_split_scan[j + 1] / tensor_split_scan.back();
            ne[split_dim] = high - low;
        }

        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = tensor->nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->data   = (char *) ggml_backend_buffer_get_base(simple_buf)
            + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        t_ij->extra = tensor->extra;
        t_ij->view_offs = tensor->view_offs;
        t_ij->view_src = tensor->view_src;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src->buffer, tensor->view_src, j);
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i]->buffer, tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }
    buf_ctx->simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) buffer->context;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_fix_via_sync =*/ false);
    std::vector<ggml_tensor *> simple_tensors;
    {
        auto it = buf_ctx->simple_tensors.find(tensor);
        assert(it != buf_ctx->simple_tensors.end());
        simple_tensors = it->second;
    }

    switch (split_state) {
        case GGML_BACKEND_SPLIT_STATE_BY_NE0: {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            const size_t row_size_full = ggml_row_size(tensor->type, tensor->ne[0]);
            size_t row_offset_j = 0;
            for (ggml_tensor * t : simple_tensors) {
                const size_t row_size_j = ggml_row_size(tensor->type, t->ne[0]);
                for (int64_t i1 = 0; i1 < tensor->ne[1]; i1++) {
                    ggml_backend_tensor_set(t, (const char *) data + i1*row_size_full + row_offset_j, i1*row_size_j, row_size_j);
                }
                row_offset_j += row_size_j;
            }
            GGML_ASSERT(row_offset_j == row_size_full);
        } break;
        case GGML_BACKEND_SPLIT_STATE_BY_NE1: {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            size_t data_offset_j = 0;
            for (ggml_tensor * t : simple_tensors) {
                const size_t nbytes_j = ggml_nbytes(t);
                ggml_backend_tensor_set(t, (const char *) data + data_offset_j, 0, nbytes_j);
                data_offset_j += nbytes_j;
            }
            GGML_ASSERT(data_offset_j == size);
        } break;
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

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_fix_via_sync =*/ false);
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

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ nullptr,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(ggml_backend_buffer_t buf, const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buffer_type_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context;
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer(ggml_backend_meta_buffer_type_simple_buft(buft, i), size);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(ggml_init(params), simple_buf, 1);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph cgraph;
        int         offset;

        ggml_tensor * tmp = nullptr;

        cgraph_config(ggml_cgraph cgraph, int offset) : cgraph(cgraph), offset(offset) {}
    };
    struct backend_config {
        ggml_backend_t             backend;
        std::vector<cgraph_config> cgraphs;
        std::vector<ggml_tensor *> nodes;

        ggml_context          * ctx = nullptr;
        ggml_backend_buffer_t   buf = nullptr;

        backend_config(ggml_backend_t backend) : backend(backend) {}
    };
    std::vector<backend_config> backend_configs;
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    return "Meta";

    GGML_UNUSED(backend);
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_tensor_set_async(
            ggml_backend_meta_simple_backend(backend, i), ggml_backend_meta_buffer_simple_tensor(tensor->buffer, tensor, i), data, offset, size);
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(n_backends >= 1);
    ggml_backend_tensor_get_async( // TODO other backends may be more optimal
        ggml_backend_meta_simple_backend(backend, 0), ggml_backend_meta_buffer_simple_tensor(tensor->buffer, tensor, 0), data, offset, size);
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
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

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        bcj.cgraphs.clear();
        bcj.nodes.clear();
        bcj.nodes.reserve(2*cgraph->n_nodes);

        for (int i = 0; i < cgraph->n_nodes; i++) {
            bcj.nodes.push_back(ggml_backend_meta_buffer_simple_tensor(cgraph->nodes[i]->buffer, cgraph->nodes[i], j));
            GGML_ASSERT(bcj.nodes[i]);
        }
    }

    size_t max_tmp_size = 0;
    {
        int i_start = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            bool new_subgraph = false;
            if (i + 1 == cgraph->n_nodes) {
                new_subgraph = true;
            } else {
                const ggml_backend_meta_split_state split_state_unfixed = ggml_backend_meta_get_split_state(node, false);
                const ggml_backend_meta_split_state split_state_fixed   = ggml_backend_meta_get_split_state(node, true);
                // GGML_ASSERT(split_state_unfixed == GGML_BACKEND_SPLIT_STATE_PARTIAL);
                // GGML_ASSERT(split_state_fixed   == GGML_BACKEND_SPLIT_STATE_MIRRORED);
                new_subgraph = split_state_fixed != split_state_unfixed;
            }
            if (!new_subgraph) {
                continue;
            }

            for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                bcj.cgraphs.emplace_back(*cgraph, i_start);
                bcj.nodes.insert(bcj.nodes.begin() + i_start + bcj.cgraphs.size()-1, nullptr);
                bcj.cgraphs.back().cgraph.nodes = bcj.nodes.data() + i_start + bcj.cgraphs.size()-1;
                bcj.cgraphs.back().cgraph.n_nodes = i + 1 - i_start + 1;
            }
            max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
            i_start = i + 1;
        }
    }
    const size_t n_subgraphs = backend_ctx->backend_configs[0].cgraphs.size();

    ggml_init_params params = {
        /*.mem_size   =*/ n_subgraphs*2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        if (bcj.ctx != nullptr) {
            ggml_free(bcj.ctx);
        }
        bcj.ctx = ggml_init(params);
        if (bcj.buf != nullptr) {
            ggml_backend_buffer_free(bcj.buf);
        }
        bcj.buf = ggml_backend_alloc_buffer(bcj.backend, max_tmp_size);
    }

    for (size_t i = 0; i < n_subgraphs; i++) {
        if (i > 0) {
            for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_tensor * node = bcj.cgraphs[i-1].cgraph.nodes[bcj.cgraphs[i-1].cgraph.n_nodes-1];
                bcj.cgraphs[i].tmp = ggml_dup_tensor(bcj.ctx, node);
                bcj.cgraphs[i].tmp->buffer = bcj.buf;
                bcj.cgraphs[i].tmp->data = ggml_backend_buffer_get_base(bcj.buf);
                for (size_t offset_j = 1; offset_j < backend_ctx->backend_configs.size(); offset_j *= 2) {
                    const size_t j_other = j ^ offset_j;
                    auto & bcj_other = backend_ctx->backend_configs[j_other];
                    ggml_tensor * node_other = bcj_other.cgraphs[i-1].cgraph.nodes[bcj_other.cgraphs[i-1].cgraph.n_nodes-1];
                    ggml_backend_tensor_copy_async(bcj_other.backend, bcj.backend, node_other, bcj.cgraphs[i].tmp);
                    bcj.cgraphs[i].cgraph.nodes[0] = ggml_add_inplace(bcj.ctx, node, bcj.cgraphs[i].tmp);
                }
                ggml_backend_synchronize(bcj.backend);
            }
        }
        for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            while (bcj.cgraphs[i].cgraph.nodes[0] == nullptr) {
                bcj.cgraphs[i].cgraph.nodes   += 1;
                bcj.cgraphs[i].cgraph.n_nodes -= 1;
                GGML_ASSERT(bcj.cgraphs[i].cgraph.n_nodes > 0);
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, &bcj.cgraphs[i].cgraph);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            ggml_backend_synchronize(bcj.backend);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
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
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    const size_t n_devs = ggml_backend_meta_device_n_devs(dev);

    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context;
    backend_ctx->backend_configs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        backend_ctx->backend_configs.emplace_back(
            ggml_backend_dev_init(ggml_backend_meta_device_simple_dev(dev, i), params));
    }

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

enum ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_fix_via_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            return dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
        }

        std::vector<ggml_backend_meta_split_state> src_split_states(GGML_MAX_SRC, GGML_BACKEND_SPLIT_STATE_NONE);
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_split_states[i] = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
                continue;
            }
            src_split_states[i] = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_fix_via_sync =*/ true);
        }

        if (tensor->op == GGML_OP_MUL_MAT) {
            if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_MIRRORED && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
                return GGML_BACKEND_SPLIT_STATE_MIRRORED;
            }
            if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE1 && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
                return GGML_BACKEND_SPLIT_STATE_BY_NE0;
            }
            if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE0 && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_BY_NE0) {
                return GGML_BACKEND_SPLIT_STATE_PARTIAL;
            }
            return assume_fix_via_sync ? GGML_BACKEND_SPLIT_STATE_MIRRORED : GGML_BACKEND_SPLIT_STATE_UNKNOWN;
        }

        ggml_backend_meta_split_state homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_NONE;
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (homogeneous_src_split_state == GGML_BACKEND_SPLIT_STATE_NONE) {
                homogeneous_src_split_state = src_split_states[i];
            } else if (src_split_states[i] != homogeneous_src_split_state) {
                homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
            }
        }
        if (homogeneous_src_split_state == GGML_BACKEND_SPLIT_STATE_NONE) {
            homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_MIRRORED;
        }
        return homogeneous_src_split_state;
    };

    if (buf_ctx->split_state_cache.find(tensor) == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[tensor] = calculate_split_state();
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[tensor];
    GGML_ASSERT(ret != GGML_BACKEND_SPLIT_STATE_NONE);
    if (assume_fix_via_sync && ret == GGML_BACKEND_SPLIT_STATE_UNKNOWN) {
        ret = GGML_BACKEND_SPLIT_STATE_MIRRORED;
    }
    return ret;
}
