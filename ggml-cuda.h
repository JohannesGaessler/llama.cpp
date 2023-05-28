#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

struct ggml_tensor_extra_gpu {
    int layer; // which layer the tensor is on
    int i_device; // which device the data is on
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
};

void   ggml_init_cublas(void);
void ggml_cuda_set_tensor_split(float * tensor_split);

void   ggml_cuda_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cuda_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

// TODO: export these with GGML_API
void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

void ggml_cuda_load_data(const char * fname, struct ggml_tensor * tensors, size_t offset, int n_layer);
void ggml_cuda_free_data(struct ggml_tensor * tensor);
bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
