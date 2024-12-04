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

struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
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
    {GGUF_TYPE_STRING,  sizeof(struct gguf_str)},
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

union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    // bool     bool_; // stored as int8 instead

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

struct gguf_header {
    char magic[4];

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_tensor_info {
    struct ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context {
    struct gguf_header header;

    std::vector<struct gguf_kv> kv;
    struct gguf_tensor_info * info;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

static size_t gguf_type_size(enum gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents integer overflow
    if (p->n >= SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\n", __func__, p->n);
        return false;
    }

    p->data = (char *) calloc(p->n + 1, 1);
    if (!p->data) {
        fprintf(stderr, "%s: failed to allocate memory for string of length %" PRIu64 "\n", __func__, p->n);
        return false;
    }

    ok = ok && gguf_fread_el(file, p->data, p->n, offset);

    return ok;
}

static void gguf_free_kv(struct gguf_kv * kv) {
    if (kv->key.data) {
        free(kv->key.data);
    }

    if (kv->type == GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
            free(kv->value.str.data);
        }
    }

    if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
            if (kv->value.arr.type == GGUF_TYPE_STRING) {
                for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                    struct gguf_str * str = &((struct gguf_str *) kv->value.arr.data)[j];
                    if (str->data) {
                        free(str->data);
                    }
                }
            }
            free(kv->value.arr.data);
        }
    }
}

struct gguf_context * gguf_init_empty(void) {
    if (sizeof(float)  != 4) {
        GGML_ABORT("support for floats with != 32 bits not implemented");
    }
    if (sizeof(double) != 8) {
        GGML_ABORT("support for doubles with != 64 bits not implemented");
    }
    struct gguf_context * ctx = (struct gguf_context *) calloc(1, sizeof(struct gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        return NULL;
    }

    memcpy(ctx->header.magic, GGUF_MAGIC, sizeof(ctx->header.magic));
    ctx->header.version   = GGUF_VERSION;
    ctx->header.n_tensors = 0;
    ctx->header.n_kv      = 0;

    ctx->info = NULL;

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->offset    = 0;
    ctx->size      = 0;

    ctx->data = NULL;

    return ctx;
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    if (sizeof(float)  != 4) {
        GGML_ABORT("support for floats with != 32 bits not implemented");
    }
    if (sizeof(double) != 8) {
        GGML_ABORT("support for doubles with != 64 bits not implemented");
    }
    FILE * file = ggml_fopen(fname, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    // offset from start of file
    size_t offset = 0;

    char magic[4];

    // check the magic before making allocations
    {
        gguf_fread_el(file, &magic, sizeof(magic), &offset);

        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return NULL;
            }
        }
    }

    bool ok = true;

    struct gguf_context * ctx = (struct gguf_context *) calloc(1, sizeof(struct gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        fclose(file);
        return NULL;
    }

    // read the header
    {
        strncpy(ctx->header.magic, magic, 4);

        ctx->info = NULL;
        ctx->data = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        // sanity checks to prevent integer/buffer overflows

        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(struct gguf_tensor_info));
        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/ggml_tensor_overhead());
        ok = ok && (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the KV pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;
        ctx->kv.resize(n_kv);

        for (uint64_t i = 0; ok && i < n_kv; ++i) {
            struct gguf_kv & kv = ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv.key, &offset);
            {
                int32_t tmp = -1; // always read enums as int32 regardless of platform
                ok = ok && gguf_fread_el(file, &tmp, sizeof(tmp), &offset);
                kv.type = gguf_type(tmp);
            }

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv.type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv.value.uint8,   sizeof(kv.value.uint8),   &offset); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv.value.int8,    sizeof(kv.value.int8),    &offset); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv.value.uint16,  sizeof(kv.value.uint16),  &offset); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv.value.int16,   sizeof(kv.value.int16),   &offset); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv.value.uint32,  sizeof(kv.value.uint32),  &offset); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv.value.int32,   sizeof(kv.value.int32),   &offset); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv.value.float32, sizeof(kv.value.float32), &offset); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv.value.uint64,  sizeof(kv.value.uint64),  &offset); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv.value.int64,   sizeof(kv.value.int64),   &offset); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv.value.float64, sizeof(kv.value.float64), &offset); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv.value.int8,    sizeof(kv.value.int8),    &offset); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv.value.str,                               &offset); break;
                case GGUF_TYPE_ARRAY:
                    {
                        {
                            int32_t tmp = -1; // always read enums as int32 regardless of platform
                            ok = ok && gguf_fread_el(file, &tmp, sizeof(tmp), &offset);
                            kv.value.arr.type = gguf_type(tmp);
                        }
                        ok = ok && gguf_fread_el(file, &kv.value.arr.n, sizeof(kv.value.arr.n), &offset);

                        switch (kv.value.arr.type) {
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
                                    // prevent integer overflow in the calloc below
                                    if (kv.value.arr.n >= SIZE_MAX/gguf_type_size(kv.value.arr.type)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv.value.arr.n);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv.value.arr.data = calloc(kv.value.arr.n, gguf_type_size(kv.value.arr.type));
                                    if (!kv.value.arr.data) {
                                        fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    ok = ok && gguf_fread_el(file, kv.value.arr.data, kv.value.arr.n * gguf_type_size(kv.value.arr.type), &offset);
                                } break;
                            case GGUF_TYPE_STRING:
                                {
                                    // prevent integer overflow in the calloc below
                                    if (kv.value.arr.n >= SIZE_MAX/sizeof(struct gguf_str)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv.value.arr.n);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv.value.arr.data = calloc(kv.value.arr.n, sizeof(struct gguf_str));
                                    if (!kv.value.arr.data) {
                                        fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    for (uint64_t j = 0; ok && j < kv.value.arr.n; ++j) {
                                        ok = ok && gguf_fread_str(file, &((struct gguf_str *) kv.value.arr.data)[j], &offset);
                                    }
                                } break;
                            case GGUF_TYPE_ARRAY:
                            default:
                                {
                                    fprintf(stderr, "%s: invalid array type %d\n", __func__, kv.value.arr.type);
                                    ok = false;
                                } break;
                        }
                    } break;
                default:
                    {
                        fprintf(stderr, "%s: invalid type %d\n", __func__, kv.type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the tensor info
    if (ctx->header.n_tensors > 0) {
        ctx->info = (struct gguf_tensor_info *) calloc(ctx->header.n_tensors, sizeof(struct gguf_tensor_info));
        if (!ctx->info) {
            fprintf(stderr, "%s: failed to allocate memory for tensor info\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        for (uint64_t i = 0; ok && i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->info[i];

            // tensor name
            {
                uint64_t n = -1;
                ok = ok && gguf_fread_el(file, &n, sizeof(n), &offset);
                if (n >= GGML_MAX_NAME) {
                    fprintf(stderr, "%s: tensor name %" PRIu64 " is too long: %" PRIu64 " >= %d\n", __func__, i, n, GGML_MAX_NAME);
                    ok = false;
                    break;
                }
                // the memory was cleared so the copied tensor name is guranteed to be null-terminated
                ok = ok && gguf_fread_el(file, info->t.name, n, &offset);

                // make sure there are no duplicated tensor names
                for (uint64_t j = 0; ok && j < i; ++j) {
                    if (strcmp(info->t.name, ctx->info[j].t.name) == 0) {
                        fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->t.name);
                        ok = false;
                        break;
                    }
                }
            }

            // tensor shape
            {
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    info->t.ne[j] = 1;
                }

                uint32_t n_dims = -1;
                ok = ok && gguf_fread_el(file, &n_dims, sizeof(n_dims), &offset);
                if (n_dims > GGML_MAX_DIMS) {
                    fprintf(stderr, "%s: tensor '%s' has invalid number of dimensions (%" PRIu32 ")\n", __func__, info->t.name, n_dims);
                    ok = false;
                    break;
                }

                ok = ok && gguf_fread_el(file, info->t.ne, n_dims*sizeof(info->t.ne[0]), &offset);

                // check that all ne are non-negative
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    if (info->t.ne[j] < 0) {
                        fprintf(stderr, "%s: tensor '%s' has invalid number of elements (%" PRIi64 ")\n",
                            __func__, info->t.name, info->t.ne[j]);
                        ok = false;
                        break;
                    }
                }

                // check that the total number of elements is representable
                if ((INT64_MAX/info->t.ne[1] <= info->t.ne[0]) ||
                    (INT64_MAX/info->t.ne[2] <= info->t.ne[0]*info->t.ne[1]) ||
                    (INT64_MAX/info->t.ne[3] <= info->t.ne[0]*info->t.ne[1]*info->t.ne[2])) {

                    fprintf(stderr, "%s: total number of elements in tensor '%s' with shape "
                        "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                        __func__, info->t.name, info->t.ne[0], info->t.ne[1], info->t.ne[2], info->t.ne[3], INT64_MAX);
                    ok = false;
                    break;
                }
            }

            // tensor type
            {
                {
                    int32_t tmp = -1; // always read enums as int32 regardless of platform
                    ok = ok && gguf_fread_el(file, &tmp, sizeof(tmp), &offset);
                    info->t.type = ggml_type(tmp);
                }

                // check that tensor type is within defined range
                if (info->t.type < 0 || info->t.type >= GGML_TYPE_COUNT) {
                    fprintf(stderr, "%s: tensor '%s' has invalid ggml type %d (%s)\n",
                        __func__, info->t.name, info->t.type, ggml_type_name(info->t.type));
                    ok = false;
                    break;
                }
                const size_t  type_size = ggml_type_size(info->t.type);
                const int64_t blck_size = ggml_blck_size(info->t.type);

                // check that row size is divisible by block size
                if (blck_size == 0 || info->t.ne[0] % blck_size != 0) {
                    fprintf(stderr, "%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                        "not a multiple of block size (%" PRId64 ")\n",
                        __func__, info->t.name, (int) info->t.type, ggml_type_name(info->t.type), info->t.ne[0], blck_size);
                    ok = false;
                    break;
                }

                // calculate byte offsets given the tensor shape and type
                info->t.nb[0] = type_size;
                info->t.nb[1] = info->t.nb[0]*(info->t.ne[0]/blck_size);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    info->t.nb[j] = info->t.nb[j - 1]*info->t.ne[j - 1];
                }
            }

            // tensor data offset within buffer
            ok = ok && gguf_fread_el(file, &info->offset, sizeof(info->offset), &offset);
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read tensor info\n", __func__);
            fclose(file);
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
        const size_t offset_align_overshoot = offset % ctx->alignment; // bytes beyond last aligned address

        if (offset_align_overshoot != 0) {
            offset += ctx->alignment - offset_align_overshoot;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->info[i];

            ctx->size += GGML_PAD(ggml_nbytes(&info->t), ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        if (*params.ctx == NULL) {
            fprintf(stderr, "%s: failed to initialize context\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = NULL;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && gguf_fread_el(file, data->data, ctx->size, &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                fclose(file);
                ggml_free(ctx_data);
                gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
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
            fclose(file);
            ggml_free(ctx_data);
            gguf_free(ctx);
            return NULL;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    fclose(file);

    return ctx;
}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    // free string memory - not great..
    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        gguf_free_kv(&ctx->kv[i]);
    }


    if (ctx->info) {
        free(ctx->info);
    }

    free(ctx);
}

const char * gguf_type_name(enum gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    return it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

int gguf_get_version(const struct gguf_context * ctx) {
    return ctx->header.version;
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
    return ctx->header.n_kv;
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
    return ctx->kv[key_id].key.data;
}

enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].type;
}

enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
}

const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
}

const char * gguf_get_arr_str(const struct gguf_context * ctx, int key_id, int i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    const struct gguf_kv & kv = ctx->kv[key_id];
    struct gguf_str * str = &((struct gguf_str *) kv.value.arr.data)[i];
    return str->data;
}

int gguf_get_arr_n(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
}

uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
}

int8_t gguf_get_val_i8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
}

uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
}

int16_t gguf_get_val_i16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

int32_t gguf_get_val_i32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
}

float gguf_get_val_f32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
}

uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
}

int64_t gguf_get_val_i64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
}

double gguf_get_val_f64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
}

bool gguf_get_val_bool(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.int8 != 0;
}

const char * gguf_get_val_str(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
}

const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type != GGUF_TYPE_ARRAY);
    GGML_ASSERT(ctx->kv[key_id].type != GGUF_TYPE_STRING);
    return &ctx->kv[key_id].value;
}

int gguf_get_n_tensors(const struct gguf_context * ctx) {
    return ctx->header.n_tensors;
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

// returns the index
static int gguf_get_or_add_key(struct gguf_context * ctx, const char * key) {
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
        return idx;
    }

    const int n_kv = gguf_get_n_kv(ctx);

    ctx->kv.resize(n_kv + 1);
    memset(&ctx->kv[n_kv], 0, sizeof(struct gguf_kv));
    ctx->kv[n_kv].key.n    = strlen(key);
    ctx->kv[n_kv].key.data = strdup(key);
    ctx->header.n_kv++;

    return n_kv;
}

void gguf_remove_key(struct gguf_context * ctx, const char * key) {
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
        gguf_free_kv(&ctx->kv[idx]);
        ctx->kv.erase(ctx->kv.begin() + idx);
        ctx->header.n_kv--;
    }
}

void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_UINT8;
    ctx->kv[idx].value.uint8 = val;
}

void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type       = GGUF_TYPE_INT8;
    ctx->kv[idx].value.int8 = val;
}

void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT16;
    ctx->kv[idx].value.uint16 = val;
}

void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT16;
    ctx->kv[idx].value.int16 = val;
}

void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT32;
    ctx->kv[idx].value.uint32 = val;
}

void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT32;
    ctx->kv[idx].value.int32 = val;
}

void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = GGUF_TYPE_FLOAT32;
    ctx->kv[idx].value.float32 = val;
}

void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT64;
    ctx->kv[idx].value.uint64 = val;
}

void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT64;
    ctx->kv[idx].value.int64 = val;
}

void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = GGUF_TYPE_FLOAT64;
    ctx->kv[idx].value.float64 = val;
}

void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type       = GGUF_TYPE_BOOL;
    ctx->kv[idx].value.int8 = val ? 1 : 0;
}

void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = GGUF_TYPE_STRING;
    ctx->kv[idx].value.str.n    = strlen(val);
    ctx->kv[idx].value.str.data = strdup(val);
}

void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);
    const size_t nbytes = n * gguf_type_size(type);

    ctx->kv[idx].type           = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = type;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = realloc(ctx->kv[idx].value.arr.data, nbytes);
    GGML_ASSERT(ctx->kv[idx].value.arr.data); // detect potential memory leak
    memcpy(ctx->kv[idx].value.arr.data, data, nbytes);
}

void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);
    const size_t nbytes = n * gguf_type_size(GGUF_TYPE_STRING);

    ctx->kv[idx].type           = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = GGUF_TYPE_STRING;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = realloc(ctx->kv[idx].value.arr.data, nbytes);
    GGML_ASSERT(ctx->kv[idx].value.arr.data); // detect potential memory leak
    for (int i = 0; i < n; ++i) {
        struct gguf_str * str = &((struct gguf_str *)ctx->kv[idx].value.arr.data)[i];
        str->n    = strlen(data[i]);
        str->data = strdup(data[i]);
        GGML_ASSERT(str->data);
    }
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src) {
    for (uint64_t i = 0; i < src->header.n_kv; ++i) {
        switch (src->kv[i].type) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, src->kv[i].key.data, src->kv[i].value.uint8);    break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, src->kv[i].key.data, src->kv[i].value.int8);     break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, src->kv[i].key.data, src->kv[i].value.uint16);   break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, src->kv[i].key.data, src->kv[i].value.int16);    break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, src->kv[i].key.data, src->kv[i].value.uint32);   break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, src->kv[i].key.data, src->kv[i].value.int32);    break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, src->kv[i].key.data, src->kv[i].value.float32);  break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, src->kv[i].key.data, src->kv[i].value.uint64);   break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, src->kv[i].key.data, src->kv[i].value.int64);    break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, src->kv[i].key.data, src->kv[i].value.float64);  break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, src->kv[i].key.data, src->kv[i].value.int8);     break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (ctx, src->kv[i].key.data, src->kv[i].value.str.data); break;
            case GGUF_TYPE_ARRAY:
                {
                    if (src->kv[i].value.arr.type == GGUF_TYPE_STRING) {
                        const char ** data = (const char **) calloc(src->kv[i].value.arr.n, sizeof(char *));
                        for (uint64_t j = 0; j < src->kv[i].value.arr.n; ++j) {
                            data[j] = ((struct gguf_str *)src->kv[i].value.arr.data)[j].data;
                        }
                        gguf_set_arr_str(ctx, src->kv[i].key.data, data, src->kv[i].value.arr.n);
                        free((void *)data);
                    } else if (src->kv[i].value.arr.type == GGUF_TYPE_ARRAY) {
                        GGML_ABORT("nested arrays not supported");
                    } else {
                        gguf_set_arr_data(ctx, src->kv[i].key.data, src->kv[i].value.arr.type,
                            src->kv[i].value.arr.data, src->kv[i].value.arr.n);
                    }
                } break;
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

    const uint64_t idx = ctx->header.n_tensors;
    ctx->info = (struct gguf_tensor_info *) realloc(ctx->info, (idx + 1)*sizeof(struct gguf_tensor_info));
    GGML_ASSERT(ctx->info); // detect potential memory leak
    ctx->info[idx].t = *tensor;
    ctx->info[idx].offset = idx == 0 ? 0 :
        ctx->info[idx - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[idx - 1].t), ctx->alignment);

    ctx->header.n_tensors++;
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
    for (uint64_t i = idx + 1; i < ctx->header.n_tensors; ++i) {
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

static void gguf_bwrite_str(struct gguf_buf * buf, const struct gguf_str * val) {
    gguf_buf_grow(buf, sizeof(val->n) + val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, &val->n, sizeof(val->n));
    }
    buf->offset += sizeof(val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val->data, val->n);
    }
    buf->offset += val->n;
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

static void gguf_write_to_buf(const struct gguf_context * ctx, struct gguf_buf * buf, bool only_meta) {
    // write header
    gguf_bwrite_el(buf, &ctx->header.magic,     sizeof(ctx->header.magic));
    gguf_bwrite_el(buf, &ctx->header.version,   sizeof(ctx->header.version));
    gguf_bwrite_el(buf, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors));
    gguf_bwrite_el(buf, &ctx->header.n_kv,      sizeof(ctx->header.n_kv));

    // write key-value pairs
    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        const struct gguf_kv & kv = ctx->kv[i];

        gguf_bwrite_str(buf, &kv.key);
        {
            const int32_t tmp = kv.type; // always write enums as int32 regardless of platform
            gguf_bwrite_el(buf, &tmp, sizeof(tmp));
        }

        switch (kv.type) {
            case GGUF_TYPE_UINT8:   gguf_bwrite_el( buf, &kv.value.uint8,   sizeof(kv.value.uint8)  ); break;
            case GGUF_TYPE_INT8:    gguf_bwrite_el (buf, &kv.value.int8,    sizeof(kv.value.int8)   ); break;
            case GGUF_TYPE_UINT16:  gguf_bwrite_el (buf, &kv.value.uint16,  sizeof(kv.value.uint16) ); break;
            case GGUF_TYPE_INT16:   gguf_bwrite_el (buf, &kv.value.int16,   sizeof(kv.value.int16)  ); break;
            case GGUF_TYPE_UINT32:  gguf_bwrite_el (buf, &kv.value.uint32,  sizeof(kv.value.uint32) ); break;
            case GGUF_TYPE_INT32:   gguf_bwrite_el (buf, &kv.value.int32,   sizeof(kv.value.int32)  ); break;
            case GGUF_TYPE_FLOAT32: gguf_bwrite_el (buf, &kv.value.float32, sizeof(kv.value.float32)); break;
            case GGUF_TYPE_UINT64:  gguf_bwrite_el (buf, &kv.value.uint64,  sizeof(kv.value.uint64) ); break;
            case GGUF_TYPE_INT64:   gguf_bwrite_el (buf, &kv.value.int64,   sizeof(kv.value.int64)  ); break;
            case GGUF_TYPE_FLOAT64: gguf_bwrite_el (buf, &kv.value.float64, sizeof(kv.value.float64)); break;
            case GGUF_TYPE_BOOL:    gguf_bwrite_el (buf, &kv.value.int8,    sizeof(kv.value.int8)   ); break;
            case GGUF_TYPE_STRING:  gguf_bwrite_str(buf, &kv.value.str                              ); break;
            case GGUF_TYPE_ARRAY:
                {
                    {
                        const int32_t tmp = kv.value.arr.type; // always write enums as int32 regardless of platform
                        gguf_bwrite_el(buf, &tmp, sizeof(tmp));
                    }
                    gguf_bwrite_el(buf, &kv.value.arr.n, sizeof(kv.value.arr.n));

                    switch (kv.value.arr.type) {
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
                                gguf_bwrite_el(buf, kv.value.arr.data, kv.value.arr.n * gguf_type_size(kv.value.arr.type));
                            } break;
                        case GGUF_TYPE_STRING:
                            {
                                for (uint64_t j = 0; j < kv.value.arr.n; ++j) {
                                    gguf_bwrite_str(buf, &((struct gguf_str *) kv.value.arr.data)[j]);
                                }
                            } break;
                        case GGUF_TYPE_ARRAY:
                        default: GGML_ABORT("invalid type");
                    }
                } break;
            default: GGML_ABORT("invalid type");
        }
    }

    // write tensor info
    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info * info = &ctx->info[i];

        struct gguf_str name = {
            /*n    =*/ strlen(info->t.name),
            /*data =*/        info->t.name,
        };
        gguf_bwrite_str(buf, &name);

        const uint32_t n_dims = ggml_n_dims(&info->t);
        gguf_bwrite_el(buf, &n_dims, sizeof(n_dims));

        for (uint32_t j = 0; j < n_dims; ++j) {
            gguf_bwrite_el(buf, &info->t.ne[j], sizeof(info->t.ne[j]));
        }
        {
            const int32_t tmp = info->t.type; // always write enums as int32 regardless of platform
            gguf_bwrite_el(buf, &tmp, sizeof(tmp));
        }
        gguf_bwrite_el(buf, &info->offset, sizeof(info->offset));
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset     = buf->offset;
        const size_t offset_pad = GGML_PAD(offset, ctx->alignment);

        if (offset_pad != offset) {
            uint8_t pad = 0;
            for (size_t i = 0; i < offset_pad - offset; ++i) {
                gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }
    }

    if (only_meta) {
        return;
    }

    size_t offset = 0;

    // write tensor data
    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info * info = &ctx->info[i];

        const size_t size     = ggml_nbytes(&info->t);
        const size_t size_pad = GGML_PAD(size, ctx->alignment);

        gguf_bwrite_tensor_data(buf, &info->t);

        const uint8_t pad = 0;
        for (size_t j = size; j < size_pad; ++j) {
            gguf_bwrite_el(buf, &pad, sizeof(pad));
        }

        GGML_ASSERT(offset == info->offset);

        offset += size_pad;
    }
}

void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = ggml_fopen(fname, "wb");
    if (!file) {
        GGML_ABORT("failed to open file for writing");
    }

    struct gguf_buf buf = gguf_buf_init(16*1024);

    gguf_write_to_buf(ctx, &buf, only_meta);

    fwrite(buf.data, 1, buf.offset, file); // buf.offset == number of bytes that are in use

    gguf_buf_free(buf);

    fclose(file);
}

size_t gguf_get_meta_size(const struct gguf_context * ctx) {
    // no allocs - only compute size
    struct gguf_buf buf = gguf_buf_init(0);

    gguf_write_to_buf(ctx, &buf, /*only_meta =*/ true);

    return buf.offset;
}

void gguf_get_meta_data(const struct gguf_context * ctx, void * data) {
    struct gguf_buf buf = gguf_buf_init(16*1024);

    gguf_write_to_buf(ctx, &buf, /*only_meta =*/ true);

    memcpy(data, buf.data, buf.offset);

    gguf_buf_free(buf);
}
