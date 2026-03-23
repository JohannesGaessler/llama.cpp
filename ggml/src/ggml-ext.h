#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// This is a "staging" header for new ggml API
// It is not publicly available and it should not be used by 3rd party projects
//
// When the API matures enough, it will be moved to the official public API

//
// Meta backend
//

#define GGML_BACKEND_META_MAX_DEVICES 16

enum ggml_backend_meta_split_axis {
    // tensor split by tensor dimensions:
    GGML_BACKEND_SPLIT_AXIS_0   =  0,
    GGML_BACKEND_SPLIT_AXIS_1   =  1,
    GGML_BACKEND_SPLIT_AXIS_2   =  2,
    GGML_BACKEND_SPLIT_AXIS_3   =  3,

    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
    GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

    // for internal bookkeeping only:
    GGML_BACKEND_SPLIT_AXIS_NONE     = 98,
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN  = 99,
};
GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

struct ggml_backend_meta_split_state {
    enum ggml_backend_meta_split_axis axis;
    int64_t                           ne[GGML_BACKEND_META_MAX_DEVICES];
};

// function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

// create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
// TODO: this looks a bit strange - a backend API creates a device. I think we should try
//       express this as a backend registry functionality instead
GGML_API ggml_backend_dev_t ggml_backend_meta_device(
    ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);
