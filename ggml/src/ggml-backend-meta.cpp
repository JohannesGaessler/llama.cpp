#include "ggml-alloc.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

// meta backend buffer type

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Meta";

    GGML_UNUSED(buft);
}

struct ggml_backend_meta_buffer_type_context {
    struct buft_config {
        ggml_backend_buffer_type_t buft;
        int64_t                    tensor_split;

        buft_config(ggml_backend_buffer_type_t buft, int64_t tensor_split) : buft(buft), tensor_split(tensor_split) {
            GGML_ASSERT(tensor_split >= 0);
        }

        bool operator<(const buft_config & other) const {
            return buft < other.buft || (buft == other.buft && tensor_split < other.tensor_split);
        }
    };
    std::vector<buft_config> buft_configs;

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return buft_configs < other.buft_configs;
    }
};

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

size_t ggml_backend_meta_buffer_type_n_bufts(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * ctx = ((const ggml_backend_meta_buffer_type_context *) buft->context);
    return ctx->buft_configs.size();
}

ggml_backend_buffer_type_t ggml_backend_meta_buffer_type_simple_buft(ggml_backend_buffer_type_t buft, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * ctx = ((const ggml_backend_meta_buffer_type_context *) buft->context);
    GGML_ASSERT(index < ctx->buft_configs.size());
    return ctx->buft_configs[index].buft;
}

void ggml_backend_meta_buffer_type_ne_bounds(ggml_backend_buffer_type_t buft, int64_t ne, int64_t * ne_bounds) {
    GGML_ASSERT(ggml_backend_buffer_type_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * ctx = ((const ggml_backend_meta_buffer_type_context *) buft->context);
    int64_t tensor_split_sum = 0;
    for (size_t i = 0; i < ctx->buft_configs.size(); i++) {
        const int64_t ts = ctx->buft_configs[i].tensor_split;
        ne_bounds[i]      = ts * ne;
        tensor_split_sum += ts;
    }
    for (size_t i = 0; i < ctx->buft_configs.size(); i++) {
        ne_bounds[i] /= tensor_split_sum;
    }
    ne_bounds[ctx->buft_configs.size()] = ne;
}

ggml_backend_buffer_type_t ggml_backend_meta_buffer_type(ggml_backend_buffer_type_t * bufts, int64_t * tensor_split, size_t n_bufts) {
    static std::vector<std::unique_ptr<ggml_backend_meta_buffer_type_context>>              ctxs;
    static std::map<ggml_backend_meta_buffer_type_context, struct ggml_backend_buffer_type> meta_bufts;

    ggml_backend_meta_buffer_type_context ctx;
    ctx.buft_configs.reserve(n_bufts);
    for (size_t i = 0; i < n_bufts; i++) {
        ctx.buft_configs.emplace_back(bufts[i], tensor_split[i]);
    }
    {
        auto it = meta_bufts.find(ctx);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_buffer_type_context>(ctx));

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_bufts.emplace(*ctxs.back(), meta_buft);
    return &result.first->second;
}

struct ggml_backend_meta_buffer_context {
    ggml_context * orig_ctx;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;
};

bool ggml_backend_meta_buffer_simple_tensors(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, struct ggml_tensor ** simple_tensors) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_bufts = ggml_backend_meta_buffer_type_n_bufts(buffer->buft);
    for (size_t j = 0; j < n_bufts; j++) {
        simple_tensors[j] = ggml_get_first_tensor(buf_ctx->buf_configs[j].ctx);
        assert(simple_tensors[j] != nullptr);
    }
    for (ggml_tensor * t = ggml_get_first_tensor(buf_ctx->orig_ctx); t != nullptr; t = ggml_get_next_tensor(buf_ctx->orig_ctx, t)) {
        if (t == tensor) {
            return true;
        }
        for (size_t j = 0; j < n_bufts; j++) {
            simple_tensors[j] = ggml_get_next_tensor(buf_ctx->buf_configs[j].ctx, simple_tensors[j]);
            assert(simple_tensors[j] != nullptr);
        }
    }
    return false;
}

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static void ggml_backend_meta_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    const size_t n_bufts = ggml_backend_meta_buffer_type_n_bufts(buffer->buft);

    std::vector<ggml_tensor *> simple_tensors(n_bufts);
    const bool success = ggml_backend_meta_buffer_simple_tensors(buffer, tensor, simple_tensors.data());
    GGML_ASSERT(success);

    constexpr size_t j_split = 1; // TODO generalize
    std::vector<int64_t> ne_bounds(n_bufts + 1);
    ggml_backend_meta_buffer_type_ne_bounds(buffer->buft, tensor->ne[j_split], ne_bounds.data());

    for (size_t j = 0; j < n_bufts; j++) {
        ggml_backend_tensor_set(simple_tensors[j], (const char *) data + ne_bounds[j]*tensor->nb[1], offset, ggml_nbytes(simple_tensors[j]));
    }
}

static void ggml_backend_meta_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    const size_t n_bufts = ggml_backend_meta_buffer_type_n_bufts(buffer->buft);

    std::vector<ggml_tensor *> simple_tensors(n_bufts);
    const bool success = ggml_backend_meta_buffer_simple_tensors(buffer, tensor, simple_tensors.data());
    GGML_ASSERT(success);

    constexpr size_t j_split = 1; // TODO generalize
    std::vector<int64_t> ne_bounds(n_bufts + 1);
    ggml_backend_meta_buffer_type_ne_bounds(buffer->buft, tensor->ne[j_split], ne_bounds.data());

    for (size_t j = 0; j < n_bufts; j++) {
        ggml_backend_tensor_get(simple_tensors[j], (char *) data + ne_bounds[j]*tensor->nb[1], offset, ggml_nbytes(simple_tensors[j]));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ nullptr,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_meta_buffer_shared_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_shared_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ nullptr,
    /* .reset           = */ nullptr,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

ggml_backend_buffer_t ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
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

    const size_t n_bufts = ggml_backend_meta_buffer_type_n_bufts(buft);
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context;
    buf_ctx->orig_ctx = ctx;
    for (size_t i = 0; i < n_bufts; i++) {
        buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    std::vector<int64_t> ne_bounds(n_bufts + 1);
    for (const ggml_tensor * t : original_tensors) {
        int64_t ne[GGML_MAX_DIMS];
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ne[k] = t->ne[k];
        }

        constexpr size_t j_split = 1; // TODO generalize
        ggml_backend_meta_buffer_type_ne_bounds(buft, t->ne[1], ne_bounds.data());
        for (size_t j = 0; j < n_bufts; j++) {
            ne[j_split] = ne_bounds[j + 1] - ne_bounds[j];
            ggml_tensor * t_j = ggml_new_tensor(buf_ctx->buf_configs[j].ctx, t->type, GGML_MAX_DIMS, ne);
            ggml_set_name(t_j, t->name);
        }
    }

    for (size_t i = 0; i < n_bufts; i++) {
        buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buffer_type_simple_buft(buft, i));
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

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
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
