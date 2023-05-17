#include "ggml.h"

#define CUFILE_CHECK(status)                                                               \
    do {                                                                                \
        CUfileError_t status_ = (status);                                                     \
        if (status_.err != CU_FILE_SUCCESS) {                                                  \
            fprintf(stderr, "cuFile error %d at %s:%d\n", status_.err, __FILE__, __LINE__);    \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_init_cublas(void);

void   ggml_cuda_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cuda_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

// TODO: export these with GGML_API
void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size);
void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

void ggml_cuda_transform_tensor(struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
