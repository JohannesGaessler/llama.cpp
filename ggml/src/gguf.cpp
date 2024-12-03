#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

template <typename T>
struct type_to_gguf_type;

template <>
struct type_to_gguf_type<uint8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
};

template <>
struct type_to_gguf_type<int8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT8;
};

template <>
struct type_to_gguf_type<uint16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
};

template <>
struct type_to_gguf_type<int16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT16;
};

template <>
struct type_to_gguf_type<uint32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
};

template <>
struct type_to_gguf_type<int32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT32;
};

template <>
struct type_to_gguf_type<float> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
struct type_to_gguf_type<bool> {
    static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
};

template <>
struct type_to_gguf_type<std::string> {
    static constexpr enum gguf_type value = GGUF_TYPE_STRING;
};

template <>
struct type_to_gguf_type<uint64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
};

template <>
struct type_to_gguf_type<int64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT64;
};

template <>
struct type_to_gguf_type<double> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   sizeof(uint8_t)},
    {GGUF_TYPE_INT8,    sizeof(int8_t)},
    {GGUF_TYPE_UINT16,  sizeof(uint16_t)},
    {GGUF_TYPE_INT16,   sizeof(int16_t)},
    {GGUF_TYPE_UINT32,  sizeof(uint32_t)},
    {GGUF_TYPE_INT32,   sizeof(int32_t)},
    {GGUF_TYPE_FLOAT32, sizeof(float)},
    {GGUF_TYPE_BOOL,    sizeof(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  sizeof(uint64_t)},
    {GGUF_TYPE_INT64,   sizeof(int64_t)},
    {GGUF_TYPE_FLOAT64, sizeof(double)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const char *> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8,   "u8"},
    {GGUF_TYPE_INT8,    "i8"},
    {GGUF_TYPE_UINT16,  "u16"},
    {GGUF_TYPE_INT16,   "i16"},
    {GGUF_TYPE_UINT32,  "u32"},
    {GGUF_TYPE_INT32,   "i32"},
    {GGUF_TYPE_FLOAT32, "f32"},
    {GGUF_TYPE_BOOL,    "bool"},
    {GGUF_TYPE_STRING,  "str"},
    {GGUF_TYPE_ARRAY,   "arr"},
    {GGUF_TYPE_UINT64,  "u64"},
    {GGUF_TYPE_INT64,   "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static size_t gguf_type_size(enum gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

struct gguf_kv {
    std::string key;

    enum gguf_type type;

    std::vector<int8_t>      arr;
    std::vector<std::string> arr_string;

    template <typename T>
    gguf_kv(const std::string & key, const T value)
            : key(key), type(type_to_gguf_type<T>::value) {
        arr.resize(sizeof(T));
        memcpy(arr.data(), &value, sizeof(T));
    }

    template <typename T>
    gguf_kv(const std::string & key, const std::vector<T> & value)
            : key(key), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        const size_t nbytes = value.size()*sizeof(T);
        arr.resize(nbytes);
        memcpy(arr.data(), value.data(), nbytes);
    }

    gguf_kv(const std::string & key, const bool value)
            : key(key), type(GGUF_TYPE_BOOL) {
        GGML_ASSERT(!key.empty());
        const int8_t tmp = value ? 1 : 0;
        arr.resize(sizeof(tmp));
        memcpy(arr.data(), &tmp, sizeof(tmp));
    }

    gguf_kv(const std::string & key, const std::string & value)
            : key(key), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        arr_string.push_back(value);
    }

    const std::string & get_key() const {
        return key;
    }

    const enum gguf_type & get_type() const {
        return type;
    }

    size_t get_ne() const {
        if (type == GGUF_TYPE_STRING) {
            return arr_string.size();
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(arr.size() % type_size == 0);
        return arr.size() / type_size;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        GGML_ASSERT(type != GGUF_TYPE_BOOL);
        GGML_ASSERT(type != GGUF_TYPE_STRING);
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(arr.size() % type_size == 0);
        GGML_ASSERT(arr.size() >= (i+1)*type_size);
        return reinterpret_cast<const T *>(arr.data())[i];
    }

    bool get_val_bool(const size_t i = 0) const {
        GGML_ASSERT(type == GGUF_TYPE_BOOL);
        GGML_ASSERT(arr.size() >= (i+1)*gguf_type_size(type));
        return reinterpret_cast<const int8_t *>(arr.data())[i] != 0;
    }

    const std::string & get_val_string(const size_t i = 0) const {
        GGML_ASSERT(type == GGUF_TYPE_STRING);
        GGML_ASSERT(arr_string.size() >= i+1);
        return arr_string[i];
    }
};

struct gguf_tensor_info {
    struct ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<struct gguf_kv> kv;
    std::vector<struct gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

struct gguf_reader {
    FILE * file;
    size_t offset = 0;

    gguf_reader(const std::string & fname) {
        file = ggml_fopen(fname.c_str(), "rb");
    }

    ~gguf_reader() {
        fclose(file);
    }

    template <typename T>
    bool read(T & dst) {
        const size_t n = fread(&dst, 1, sizeof(dst), file);
        offset += n;
        return n == sizeof(dst);
    }

    template <typename T>
    bool read(std::vector<T> & dst) {
        {
            uint64_t n = -1;
            if (!read(n)) {
                return false;
            }
            dst.resize(n);
        }
        for (size_t i = 0; i < dst.size(); ++i) {
            if (!read(dst[i])) {
                return false;
            }
        }
        return true;
    }

    bool read(bool & dst) {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type & dst) {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type & dst) {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) {
        uint64_t size = -1;
        if (!read(size)) {
            return false;
        }
        dst.resize(size);
        const size_t n = fread(dst.data(), 1, size, file);
        offset += n;
        return n == dst.length();
    }

    bool read(void * dst, const size_t size) {
        const size_t n = fread(dst, 1, size, file);
        offset += n;
        return n == size;
    }
};

struct gguf_context * gguf_init_empty(void) {
    return new gguf_context;
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    struct gguf_reader gr(fname);
    if (!gr.file) {
        fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    struct gguf_context * ctx = new gguf_context;

    // check the magic before making allocations
    {
        char magic[4];
        gr.read(magic);

        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                gguf_free(ctx);
                return NULL;
            }
        }
    }

    bool ok = true;
    uint64_t n_kv = -1;

    // read the header
    {
        ctx->data = NULL;

        ok = ok && gr.read(ctx->version);

        if (ctx->version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            gguf_free(ctx);
            return NULL;
        }
        {
            uint64_t tmp = -1;
            ok = ok && gr.read(tmp);
            ctx->info.resize(tmp); // FIXME failure handling
        }
        ok = ok && gr.read(n_kv);

        // sanity checks to prevent integer/buffer overflows

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the KV pairs
    {
        for (uint64_t i = 0; ok && i < n_kv; ++i) {
            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            std::string key;
            gguf_type   type = gguf_type(-1);

            ok = ok && gr.read(key);
            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                gguf_type type_arr = gguf_type(-1);
                ok = ok && gr.read(type_arr);

                switch (type_arr) {
                    case GGUF_TYPE_UINT8: {
                        std::vector<uint8_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_INT8: {
                        std::vector<int8_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_UINT16: {
                        std::vector<uint16_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_INT16: {
                        std::vector<int16_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_UINT32: {
                        std::vector<uint32_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_INT32: {
                        std::vector<int32_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_FLOAT32: {
                        std::vector<float> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_UINT64: {
                        std::vector<uint64_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_INT64: {
                        std::vector<int64_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_FLOAT64: {
                        std::vector<double> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_BOOL: {
                        std::vector<int8_t> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_STRING: {
                        std::vector<std::string> value;
                        if (!gr.read(value)) {
                            ok = false;
                            break;
                        }
                        ctx->kv.emplace_back(key, value);
                    } break;
                    case GGUF_TYPE_ARRAY:
                    default:
                        {
                            fprintf(stderr, "%s: invalid array type %d\n", __func__, type_arr);
                            ok = false;
                        } break;
                }
                continue;
            }

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (type) {
                case GGUF_TYPE_UINT8: {
                    uint8_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_INT8: {
                    int8_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_UINT16: {
                    uint16_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_INT16: {
                    int16_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_UINT32: {
                    uint32_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_INT32: {
                    int32_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_FLOAT32: {
                    float value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_UINT64: {
                    uint64_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_INT64: {
                    int64_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_FLOAT64: {
                    double value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_BOOL: {
                    int8_t value = -1;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value != 0);
                } break;
                case GGUF_TYPE_STRING: {
                    std::string value;
                    if (!gr.read(value)) {
                        ok = false;
                        break;
                    }
                    ctx->kv.emplace_back(key, value);
                } break;
                case GGUF_TYPE_ARRAY:
                default:
                    {
                        fprintf(stderr, "%s: invalid type %d\n", __func__, type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the tensor info
    if (!ctx->info.empty()) {
        const uint64_t n_tensors = ctx->info.size();
        for (uint64_t i = 0; ok && i < n_tensors; ++i) {
            struct gguf_tensor_info & info = ctx->info[i];

            // tensor name
            {
                std::string name;
                ok = ok && gr.read(name);
                if (name.length() >= GGML_MAX_NAME) {
                    fprintf(stderr, "%s: tensor name %" PRIu64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                    ok = false;
                    break;
                }
                ggml_set_name(&info.t, name.c_str());

                // make sure there are no duplicated tensor names
                for (uint64_t j = 0; ok && j < i; ++j) {
                    if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                        fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info.t.name);
                        ok = false;
                        break;
                    }
                }
            }

            // tensor shape
            {
                uint32_t n_dims = -1;
                ok = ok && gr.read(n_dims);
                if (n_dims > GGML_MAX_DIMS) {
                    fprintf(stderr, "%s: tensor '%s' has invalid number of dimensions (%" PRIu32 ")\n", __func__, info.t.name, n_dims);
                    ok = false;
                    break;
                }
                for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                    info.t.ne[j] = 1;
                    if (j < n_dims) {
                        ok = ok && gr.read(info.t.ne[j]);
                    }

                    // check that all ne are non-negative
                    if (info.t.ne[j] < 0) {
                        fprintf(stderr, "%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements (%" PRIi64 ")\n",
                            __func__, info.t.name, j, info.t.ne[j]);
                        ok = false;
                        break;
                    }
                }

                // check that the total number of elements is representable
                if ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                    (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                    (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2])) {

                    fprintf(stderr, "%s: total number of elements in tensor '%s' with shape "
                        "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                        __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                    ok = false;
                    break;
                }
            }

            // tensor type
            {
                ok = ok && gr.read(info.t.type);

                // check that tensor type is within defined range
                if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                    fprintf(stderr, "%s: tensor '%s' has invalid ggml type %d (%s)\n",
                        __func__, info.t.name, info.t.type, ggml_type_name(info.t.type));
                    ok = false;
                    break;
                }
                const size_t  type_size = ggml_type_size(info.t.type);
                const int64_t blck_size = ggml_blck_size(info.t.type);

                // check that row size is divisible by block size
                if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                    fprintf(stderr, "%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                        "not a multiple of block size (%" PRId64 ")\n",
                        __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                    ok = false;
                    break;
                }

                // calculate byte offsets given the tensor shape and type
                info.t.nb[0] = type_size;
                info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
                }
            }

            // tensor data offset within buffer
            ok = ok && gr.read(info.offset);
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read tensor info\n", __func__);
            gguf_free(ctx);
            return NULL;
        }
    }

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_align_overshoot = gr.offset % ctx->alignment; // bytes beyond last aligned address

        if (offset_align_overshoot != 0) {
            gr.offset += ctx->alignment - offset_align_overshoot;
            fseek(gr.file, gr.offset, SEEK_SET); // FIXME
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (const struct gguf_tensor_info & ti : ctx->info) {
            ctx->size += GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        const uint64_t n_tensors = ctx->info.size();
        const size_t mem_size =
            params.no_alloc ?
            (n_tensors    )*ggml_tensor_overhead() :
            (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        if (*params.ctx == NULL) {
            fprintf(stderr, "%s: failed to initialize context\n", __func__);
            gguf_free(ctx);
            return NULL;
        }

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = NULL;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && gr.read(data->data, ctx->size);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                ggml_free(ctx_data);
                gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint64_t i = 0; i < n_tensors; ++i) {
            struct ggml_tensor * cur = ggml_new_tensor(
                ctx_data, ctx->info[i].t.type, GGML_MAX_DIMS, ctx->info[i].t.ne);

            ok = ok && cur != NULL;

            if (!ok) {
                break;
            }

            ggml_set_name(cur, ctx->info[i].t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->info[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->info[i].offset;               // offset from data
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read the tensor data\n", __func__);
            ggml_free(ctx_data);
            gguf_free(ctx);
            return NULL;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    delete ctx;
}

const char * gguf_type_name(enum gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    return it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

int gguf_get_version(const struct gguf_context * ctx) {
    return ctx->version;
}

size_t gguf_get_alignment(const struct gguf_context * ctx) {
    return ctx->alignment;
}

size_t gguf_get_data_offset(const struct gguf_context * ctx) {
    return ctx->offset;
}

// TODO should this be a const pointer? should it exist at all?
void * gguf_get_data(const struct gguf_context * ctx) {
    return ctx->data;
}

// TODO this returns int but the underlying type is uint64
int gguf_get_n_kv(const struct gguf_context * ctx) {
    return ctx->kv.size();
}

int gguf_find_key(const struct gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].get_key().c_str();
}

enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].get_type();
}

enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].get_type();
}

const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].arr.data();
}

const char * gguf_get_arr_str(const struct gguf_context * ctx, int key_id, int i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].arr_string[i].c_str();
}

int gguf_get_arr_n(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].arr.size();
}

uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_UINT8);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_INT8);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_UINT16);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_INT16);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_UINT32);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_INT32);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int32_t>();
}

float gguf_get_val_f32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_FLOAT32);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<float>();
}

uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_UINT64);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_INT64);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int64_t>();
}

double gguf_get_val_f64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_FLOAT64);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<double>();
}

bool gguf_get_val_bool(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_BOOL);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val_bool();
}

const char * gguf_get_val_str(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_STRING);
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val_string().c_str();
}

const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_ARRAY); // FIXME
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    return ctx->kv[key_id].arr.data();
}

int gguf_get_n_tensors(const struct gguf_context * ctx) {
    return ctx->info.size();
}

int gguf_find_tensor(const struct gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensorfound = i;
            break;
        }
    }

    return tensorfound;
}

size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i) {
    return ctx->info[i].offset;
}

const char * gguf_get_tensor_name(const struct gguf_context * ctx, int i) {
    return ctx->info[i].t.name;
}

enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int i) {
    return ctx->info[i].t.type;
}

size_t gguf_get_tensor_size(const struct gguf_context * ctx, int i) {
    return ggml_nbytes(&ctx->info[i].t);
}

void gguf_remove_key(struct gguf_context * ctx, const char * key) {
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
        // gguf_free_kv(&ctx->kv[idx]);
        ctx->kv.erase(ctx->kv.begin() + idx);
    }
}

void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val) {
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n) {
    gguf_remove_key(ctx, key);

    const size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    memcpy(tmp.data(), data, nbytes);
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().type = type;
}

void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, int n) {
    gguf_remove_key(ctx, key);
    std::vector<std::string> tmp(n);
    for (int i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src) {
    const uint64_t n_kv = src->kv.size();
    for (uint64_t i = 0; i < n_kv; ++i) {
        switch (src->kv[i].get_type()) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<uint8_t>());       break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<int8_t>());        break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<uint16_t>());      break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<int16_t>());       break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<uint32_t>());      break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<int32_t>());       break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<float>());         break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<uint64_t>());      break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<int64_t>());       break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val<double>());        break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val_bool());           break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (ctx, src->kv[i].get_key().c_str(), src->kv[i].get_val_string().c_str()); break;
            // FIXME
            // case GGUF_TYPE_ARRAY:
            //     {
            //         if (src->kv[i].get_type_arr() == GGUF_TYPE_STRING) {
            //             const uint64_t n = src->kv[i].arr_string.size();
            //             const char ** data = (const char **) calloc(n, sizeof(char *));
            //             for (uint64_t j = 0; j < n; ++j) {
            //                 data[j] = src->kv[i].arr_string[i].c_str();
            //             }
            //             gguf_set_arr_str(ctx, src->kv[i].get_key().c_str(), data, n);
            //             free((void *)data);
            //         } else if (src->kv[i].get_type_arr() == GGUF_TYPE_ARRAY) {
            //             GGML_ABORT("nested arrays not supported");
            //         } else {
            //             gguf_set_arr_data(ctx, src->kv[i].get_key().c_str(), src->kv[i].get_type_arr(),
            //                 src->kv[i].arr.data(), src->kv[i].arr.size()/gguf_type_size(src->kv[i].get_type()));
            //         }
            //     } break;
            default: GGML_ABORT("invalid type");
        }
    }
}

void gguf_add_tensor(
             struct gguf_context * ctx,
        const struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    if (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicated tensor name");
    }

    struct gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type) {
    const int idx = gguf_find_tensor(ctx, name);
    if (idx < 0) {
        GGML_ABORT("tensor not found");
    }
    struct ggml_tensor * tensor = &ctx->info[idx].t;
    const size_t  type_size = ggml_type_size(type);
    const int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    const uint64_t n_tensors = ctx->info.size();
    for (uint64_t i = idx + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data) {
    const int idx = gguf_find_tensor(ctx, name);
    if (idx < 0) {
        GGML_ABORT("tensor not found");
    }

    ctx->info[idx].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

struct gguf_buf {
    void * data;
    size_t size;   // size of data
    size_t offset; // offset within data
};

static struct gguf_buf gguf_buf_init(size_t size) {
    struct gguf_buf buf = {
        /*buf.data   =*/ size == 0 ? NULL : calloc(1, size),
        /*buf.size   =*/ size,
        /*buf.offset =*/ 0,
    };

    return buf;
}

static void gguf_buf_free(struct gguf_buf buf) {
    if (buf.data) {
        free(buf.data);
    }
}

static void gguf_buf_grow(struct gguf_buf * buf, size_t size) {
    if (buf->offset + size > buf->size) {
        buf->size = 1.5f*(buf->offset + size);
        if (buf->data) {
            buf->data = realloc(buf->data, buf->size);
            GGML_ASSERT(buf->data); // detect potential memory leak
        }
    }
}

static void gguf_bwrite_el(struct gguf_buf * buf, const void * val, size_t el_size) {
    gguf_buf_grow(buf, el_size);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val, el_size);
    }
    buf->offset += el_size;
}

static void gguf_bwrite_tensor_data(struct gguf_buf * buf, const struct ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const size_t el_size = ggml_nbytes(tensor);
    gguf_buf_grow(buf, el_size);

    if (buf->data) {
        char * dst = (char *) buf->data + buf->offset;
        if (tensor->buffer) {
            ggml_backend_tensor_get(tensor, dst, 0, el_size);
        } else {
            GGML_ASSERT(tensor->data);
            memcpy(dst, tensor->data, el_size);
        }
    }
    buf->offset += el_size;
}

struct gguf_writer{
    std::vector<int8_t> data;
    bool no_alloc = false; // FIXME

    template <typename T>
    void write(const T & val) {
        const int8_t * val8 = reinterpret_cast<const int8_t *>(&val);
        for (size_t i = 0; i < sizeof(val); ++i) {
            data.push_back(val8[i]);
        }
    }

    void write(const std::vector<int8_t> & val, const enum gguf_type type) {
        GGML_ASSERT(type != GGUF_TYPE_STRING);
        GGML_ASSERT(type != GGUF_TYPE_ARRAY);
        {
            const size_t type_size = gguf_type_size(type);
            GGML_ASSERT(val.size() % type_size == 0);
            const uint64_t n = val.size() / type_size;
            write(n);
        }
        for (size_t i = 0; i < val.size(); ++i) {
            write(val[i]);
        }
    }

    void write(const bool & val) {
        const int8_t val8 = val ? 1 : 0;
        data.push_back(val8);
    }

    void write(const std::string & val) {
        {
            const uint64_t n = val.length();
            write(n);
        }

        const int8_t * val_data8 = reinterpret_cast<const int8_t *>(val.data());
        for (size_t i = 0; i < val.length(); ++i) {
            data.push_back(val_data8[i]);
        }
    }

    void write(const char * val) {
        write(std::string(val));
    }

    void write(const std::vector<std::string> & val) {
        {
            const uint64_t n = val.size();
            write(n);
        }
        for (size_t i = 0; i < val.size(); ++i) {
            write(val[i]);
        }
    }

    void write(const enum ggml_type & val) {
        const int32_t val32 = int32_t(val);
        const int8_t * val8 = reinterpret_cast<const int8_t *>(&val32);
        for (size_t i = 0; i < sizeof(val32); ++i) {
            data.push_back(val8[i]);
        }
    }

    void write(const enum gguf_type & val) {
        const int32_t val32 = int32_t(val);
        const int8_t * val8 = reinterpret_cast<const int8_t *>(&val32);
        for (size_t i = 0; i < sizeof(val32); ++i) {
            data.push_back(val8[i]);
        }
    }

    void write_tensor_data(const struct ggml_tensor & tensor) {
        GGML_ASSERT(ggml_is_contiguous(&tensor));
        const size_t nbytes = ggml_nbytes(&tensor);
        const size_t offset = data.size();
        data.resize(offset + nbytes);

        if (no_alloc) {
            return;
        }

        if (tensor.buffer) {
            ggml_backend_tensor_get(&tensor, data.data() + offset, 0, nbytes);
        } else {
            GGML_ASSERT(tensor.data);
            memcpy(data.data() + offset, tensor.data, nbytes);
        }
    }
};

static void gguf_write_to_buf(const struct gguf_context * ctx, struct gguf_writer & gw, const bool only_meta) {
    const uint64_t n_kv      = ctx->kv.size();
    const uint64_t n_tensors = ctx->info.size();

    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (uint64_t i = 0; i < n_kv; ++i) {
        const struct gguf_kv & kv = ctx->kv[i];
        const size_t ne = kv.get_ne();

        gw.write(kv.get_key());

        if (ne == 1) {
            gw.write(kv.get_type());
            switch (kv.get_type()) {
                case GGUF_TYPE_UINT8:   gw.write(kv.get_val<uint8_t>());  break;
                case GGUF_TYPE_INT8:    gw.write(kv.get_val<int8_t>());   break;
                case GGUF_TYPE_UINT16:  gw.write(kv.get_val<uint16_t>()); break;
                case GGUF_TYPE_INT16:   gw.write(kv.get_val<int16_t>());  break;
                case GGUF_TYPE_UINT32:  gw.write(kv.get_val<uint32_t>()); break;
                case GGUF_TYPE_INT32:   gw.write(kv.get_val<int32_t>());  break;
                case GGUF_TYPE_FLOAT32: gw.write(kv.get_val<float>());    break;
                case GGUF_TYPE_UINT64:  gw.write(kv.get_val<uint64_t>()); break;
                case GGUF_TYPE_INT64:   gw.write(kv.get_val<int64_t>());  break;
                case GGUF_TYPE_FLOAT64: gw.write(kv.get_val<double>());   break;
                case GGUF_TYPE_BOOL:    gw.write(kv.get_val_bool());     break;
                case GGUF_TYPE_STRING:  gw.write(kv.get_val_string());    break;
                case GGUF_TYPE_ARRAY:
                default: GGML_ABORT("invalid type");
            }
            continue;
        }

        gw.write(GGUF_TYPE_ARRAY);
        gw.write(kv.get_type());

        switch (kv.get_type()) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL:
                {
                    gw.write(kv.arr, kv.get_type());
                } break;
            case GGUF_TYPE_STRING:
                {
                    gw.write(kv.arr_string);
                } break;
            case GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }

    // write tensor info
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const struct gguf_tensor_info & info = ctx->info[i];

        gw.write(info.t.name);

        const uint32_t n_dims = ggml_n_dims(&info.t);
        gw.write(n_dims);

        for (uint32_t j = 0; j < n_dims; ++j) {
            gw.write(info.t.ne[j]);
        }
        gw.write(info.t.type);
        gw.write(info.offset);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset     = gw.data.size();
        const size_t offset_pad = GGML_PAD(offset, ctx->alignment);

        if (offset_pad != offset) {
            const int8_t pad = 0;
            for (size_t i = 0; i < offset_pad - offset; ++i) {
                gw.write(pad);
            }
        }
    }

    if (only_meta) {
        return;
    }

    size_t offset = 0;

    // write tensor data
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const struct gguf_tensor_info & info = ctx->info[i];

        const size_t size     = ggml_nbytes(&info.t);
        const size_t size_pad = GGML_PAD(size, ctx->alignment);

        gw.write_tensor_data(info.t);

        const int8_t pad = 0;
        for (size_t j = size; j < size_pad; ++j) {
            gw.write(pad);
        }

        GGML_ASSERT(offset == info.offset);

        offset += size_pad;
    }
}

void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = ggml_fopen(fname, "wb");
    if (!file) {
        GGML_ABORT("failed to open file for writing");
    }

    struct gguf_writer gw;

    gguf_write_to_buf(ctx, gw, only_meta);

    fwrite(gw.data.data(), 1, gw.data.size(), file); // buf.offset == number of bytes that are in use

    fclose(file);
}

size_t gguf_get_meta_size(const struct gguf_context * ctx) {
    // no allocs - only compute size
    struct gguf_writer gw;

    gguf_write_to_buf(ctx, gw, /*only_meta =*/ true);

    return gw.data.size();
}

void gguf_get_meta_data(const struct gguf_context * ctx, void * data) {
    struct gguf_writer gw;

    gguf_write_to_buf(ctx, gw, /*only_meta =*/ true);

    memcpy(data, gw.data.data(), gw.data.size());
}
