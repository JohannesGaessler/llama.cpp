#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;

    params.logits_all = true;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const int32_t n_ctx_train = llama_n_ctx_train(model);

    struct llama_batch batch = llama_batch_init(n_ctx_train, 0, 1);
    for (int i = 0; i < n_ctx_train; ++i) {
        common_batch_add(batch, 0, i, {0}, /*logits =*/ i == n_ctx_train-1);
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    ggml_opt_dataset_t dataset = llama_opt_dataset_init(ctx, tokens.data(), tokens.size());
    ggml_opt_context_t opt_ctx = llama_opt_init(ctx, batch);
    ggml_opt_result_t result = ggml_opt_result_init();

    ggml_opt_epoch(opt_ctx, dataset, nullptr, result, 0, nullptr, ggml_opt_epoch_callback_progress_bar);

    // struct ggml_context * c = ggml_init({1024*1024*1024, NULL, true});
    // struct ggml_tensor * a = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, 128);
    // struct ggml_tensor * b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, 512);
    // ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_cpu_buffer_type());

    // for (int i = 0; i < 512; ++i) {
    //     std::string s = common_token_to_piece(ctx, i);
    //     printf("i=%d s=%s\n", i, s.c_str());
    // }

    // for (int64_t ibatch = 0; ibatch < 10; ++ibatch) {
    //     ggml_opt_dataset_get_batch(dataset, a, b, ibatch);
    //     std::vector<llama_token> s(128);
    //     memcpy(s.data(), a->data, 128*sizeof(llama_token));
    //     std::string s2;
    //     for (llama_token t : s) {
    //         s2 += common_token_to_piece(ctx, t);
    //     }

    //     std::vector<float> l(512);
    //     memcpy(l.data(), b->data, 512*sizeof(float));
    //     std::vector<float>::iterator p = std::max_element(l.begin(), l.end());
    //     std::string s3 = common_token_to_piece(ctx, p-l.begin());

    //     printf("input='%s' output='%s'\n", s2.c_str(), s3.c_str());
    // }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
