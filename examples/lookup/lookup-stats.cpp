#include "common.h"
#include "common/common.h"
#include "common/log.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

// Min/max n-gram size to search for in prompt:
constexpr int   ngram_min =  1;
constexpr int   ngram_max =  4;
static_assert(ngram_max <= sizeof(uint64_t)/2, "A 64 bit integer can only hold information for 4 16 bit tokens.");

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    const int n_draft = params.n_draft;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    llama_set_rng_seed(ctx, params.seed);
    GGML_ASSERT(llama_n_vocab(model) < (1 << 16));

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos tgt: %d\n", add_bos);

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

    llama_ngram_cache ngram_cache;
    llama_ngram_cache ngram_cache_dynamic;
    llama_ngram_cache ngram_cache_static;
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        const int64_t t_start_draft_us = ggml_time_us();

        if (!params.lookup_cache_static.empty()) {
            try {
                ngram_cache_static = llama_ngram_cache_load(params.lookup_cache_static);
            } catch (std::system_error const &) {
                fprintf(stderr, "error: failed to open static lookup cache: %s", params.lookup_cache_static.c_str());
                exit(1);
            }
        }

        if (!params.lookup_cache_dynamic.empty()) {
            try {
                ngram_cache_dynamic = llama_ngram_cache_load(params.lookup_cache_dynamic);
            } catch (std::system_error const &) {} // if the file does not exist it will simply be created at the end of the program
        }

        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
    }

    const int n_input = inp.size();
    const int n_ctx = params.n_ctx;

    int n_drafted = 0;
    int n_accept  = 0;

    const int64_t t_start_ms = ggml_time_ms();
    for (int i_start = 0; i_start + n_ctx < n_input; i_start += n_ctx) {
        const std::vector<llama_token> inp_slice(inp.begin() + i_start, inp.begin() + i_start + n_ctx);
        std::vector<llama_token> pseudo_output;
        pseudo_output.push_back(inp_slice[0]);

        while ((int) pseudo_output.size() < n_ctx) {
            std::vector<llama_token> draft;
            draft.push_back(pseudo_output.back());

            {
                const int64_t t_start_draft_us = ggml_time_us();
                llama_ngram_cache_draft(pseudo_output, draft, n_draft, ngram_min, ngram_max, ngram_cache, ngram_cache_dynamic, ngram_cache_static);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }

            n_drafted += draft.size() - 1;

            for (size_t j = 1; j < draft.size() && (int) pseudo_output.size() < n_ctx; ++j) {
                const llama_token ground_truth = inp_slice[pseudo_output.size()];
                const llama_token drafted = draft[j];

                if (ground_truth != drafted) {
                    break;
                }

                ++n_accept;
                pseudo_output.push_back(ground_truth);

                {
                    const int64_t t_start_draft_us = ggml_time_us();
                    // llama_sampling_probability_distribution(nullptr, ctx, nullptr, j);
                    llama_ngram_cache_update(ngram_cache, ngram_min, ngram_max, pseudo_output, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
            }
            if ((int) pseudo_output.size() < n_ctx) {
                pseudo_output.push_back(inp_slice[pseudo_output.size()]);
                {
                    const int64_t t_start_draft_us = ggml_time_us();
                    llama_ngram_cache_update(ngram_cache, ngram_min, ngram_max, pseudo_output, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
            }

            draft.erase(draft.begin());

        }
        if (i_start >= n_ctx && i_start / 100000 != (i_start - n_ctx) / 100000) {
            const int64_t t_now_ms = ggml_time_ms();
            const int64_t eta_ms   = (n_input - i_start) * (t_now_ms - t_start_ms) / i_start;
            const int64_t eta_min  = eta_ms / (60*1000);
            const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

            LOG_TEE("%d/%d done, ETA: %02ld:%02ld\n", i_start, n_input, eta_min, eta_s);
        }

        llama_ngram_cache_merge(ngram_cache_dynamic, ngram_cache);
        ngram_cache.clear();
    }

    LOG_TEE("\n");

    LOG_TEE("\n");
    LOG_TEE("n_draft      = %d\n", n_draft);
    LOG_TEE("n_predict    = %d\n", n_input - n_input % n_ctx);
    LOG_TEE("n_drafted    = %d\n", n_drafted);
    LOG_TEE("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_TEE("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_TEE("n_accept     = %d\n", n_accept);
    LOG_TEE("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
