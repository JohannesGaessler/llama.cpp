#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#define GGUF_MAGIC "GGUF"

#define GGUF_VERSION 3

#define GGUF_DEFAULT_ALIGNMENT 32

#ifdef  __cplusplus
extern "C" {
#endif

    // types that can be stored as GGUF KV data
    enum gguf_type {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
    };

    struct gguf_context;

    struct gguf_init_params {
        bool no_alloc;

        // if not NULL, create a ggml_context and allocate the tensor data in it
        struct ggml_context ** ctx;
    };

    GGML_API struct gguf_context * gguf_init_empty(void);
    GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
    //GGML_API struct gguf_context * gguf_init_from_buffer(..);

    GGML_API void gguf_free(struct gguf_context * ctx);

    GGML_API const char * gguf_type_name(enum gguf_type type);

    GGML_API int    gguf_get_version    (const struct gguf_context * ctx);
    GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
    GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
    GGML_API void * gguf_get_data       (const struct gguf_context * ctx);

    GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);
    GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);
    GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int key_id);

    GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);
    GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);

    // will abort if the wrong type is used for the key
    GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int key_id);
    GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int key_id);
    GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);
    GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);
    GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);
    GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);
    GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);
    GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);
    GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);
    GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);
    GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);
    GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);
    GGML_API const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id);
    GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);

    // get raw pointer to the first element of the array with the given key_id
    // for bool arrays, note that they are always stored as int8 on all platforms (usually this makes no difference)
    GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);

    // get ith C string from array with given key_id
    GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);

    GGML_API int            gguf_get_n_tensors    (const struct gguf_context * ctx);
    GGML_API int            gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
    GGML_API size_t         gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
    GGML_API const char *   gguf_get_tensor_name  (const struct gguf_context * ctx, int i);
    GGML_API enum ggml_type gguf_get_tensor_type  (const struct gguf_context * ctx, int i);
    GGML_API size_t         gguf_get_tensor_size  (const struct gguf_context * ctx, int i);

    // removes key if it exists
    GGML_API void gguf_remove_key(struct gguf_context * ctx, const char * key);

    // overrides existing values or adds a new one
    GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t      val);
    GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t       val);
    GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t     val);
    GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t      val);
    GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t     val);
    GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t      val);
    GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float        val);
    GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t     val);
    GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t      val);
    GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double       val);
    GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool         val);
    GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);

    // creates a new array with n elements of the given type and copies the corresponding number of bytes from data
    GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);

    // creates a new array with n strings and copies the corresponding strings from data
    GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);

    // set or add KV pairs from another context
    GGML_API void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src);

    // manage tensor info
    GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);

    // after changing a tensor's type, the offsets of all tensors with higher indices are recalculated
    //   in such a way that the tensor data remains as one contiguous block (except for padding)
    GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);

    // assumes that at least gguf_get_tensor_size bytes can be read from data
    GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data);

    // writing gguf files can be done in 2 ways:
    //
    // - write the entire gguf_context to a binary file in a single pass:
    //
    //   gguf_write_to_file(ctx, fname);
    //
    // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
    //
    //   FILE * f = fopen(fname, "wb");
    //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
    //   fwrite(f, ...);
    //   void * data = gguf_meta_get_meta_data(ctx);
    //   fseek(f, 0, SEEK_SET);
    //   fwrite(f, data, gguf_get_meta_size(ctx));
    //   free(data);
    //   fclose(f);
    //

    // write the entire context to a binary file
    GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);

    // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
    GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);

    // writes the meta data to pointer "data"
    GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);

#ifdef  __cplusplus
}
#endif
