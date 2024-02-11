#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

typedef std::unordered_map<llama_token, int>            token_hashmap;
typedef std::unordered_map<uint64_t, token_hashmap> all_token_hashmap;

// min/max n-gram size to search for in prompt
constexpr int   ngram_min =  1;
constexpr int   ngram_max =  4;

// if sample size or percentage in context are below these thresholds the draft is aborted early
constexpr float draft_min_sample_size[ngram_max] = { 2,  2,  1,  1};
constexpr float     draft_min_percent[ngram_max] = {66, 50, 50, 50};

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    // max/min n-grams size to search for in prompt
    // length of the candidate / draft sequence, if match is found
    const int n_draft = params.n_draft;

    const bool dump_kv_cache = params.dump_kv_cache;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("lookup", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    GGML_ASSERT(llama_n_vocab(model) < (1 << 16));

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos tgt: %d\n", add_bos);

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

    auto update_hashmaps = [](all_token_hashmap * atcs, const llama_token * inp_data, const int inp_size, const int nnew) -> void {
        for (int ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
            all_token_hashmap * atc = atcs + ngram_size - ngram_min;

            for (int i = inp_size - nnew + ngram_size; i < inp_size; ++i) {
                const int ngram_start = i - ngram_size;
                uint64_t ngram = inp_data[ngram_start];
                for (int j = ngram_start; j < ngram_start + ngram_size; ++j) {
                    const uint64_t ngram_part = inp_data[j];
                    ngram <<= 16;
                    ngram |= ngram_part;
                }
                const llama_token token = inp_data[i];

                all_token_hashmap::iterator token_counts_it = atc->find(ngram);
                if (token_counts_it == atc->end()) {
                    token_hashmap token_counts;
                    token_counts.emplace(token, 1);
                    atc->emplace(ngram, token_counts);
                } else {
                    token_hashmap::iterator tc_it = token_counts_it->second.find(token);
                    if (tc_it == token_counts_it->second.end()) {
                        token_counts_it->second.emplace(token, 1);
                    } else {
                        tc_it->second++;
                    }
                }
            }
        }
    };

    all_token_hashmap all_token_counts[ngram_max-ngram_min+1];

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int64_t t_draft_us = 0;

    int n_past = inp.size();

    bool has_eos = false;

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    std::vector<llama_token> draft;

    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            dump_kv_cache_view_seqs(kvc_view, 40);
        }

        // print current draft sequence
        LOG("drafted %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, draft).c_str());

        int i_dft = 0;
        while (true) {
            // sample from the target model
            llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL, i_dft);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            const std::string token_str = llama_token_to_piece(ctx, id);

            if (!params.use_color) {
                printf("%s", token_str.c_str());
            }

            if (id == llama_token_eos(model)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches the draft
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                LOG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                ++n_accept;
                ++n_past;
                ++i_dft;
                inp.push_back(id);

                if (params.use_color) {
                    // color accepted draft token
                    printf("\033[34m%s\033[0m", token_str.c_str());
                    fflush(stdout);
                }
                continue;
            }

            if (params.use_color) {
                printf("%s", token_str.c_str());
            }
            fflush(stdout);


            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            break;
        }

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

        llama_batch_clear(batch_tgt);
        llama_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

        auto get_token = [](const std::vector<llama_token> inp, const std::vector<llama_token> draft, const size_t i) -> llama_token {
            return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
        };

        // generate n_pred tokens through prompt lookup
        auto prompt_lookup = [&]() -> void {
            const int inp_size = inp.size();
            for (int ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
                all_token_counts[ngram_size-ngram_min].clear();
            }
            update_hashmaps(all_token_counts, inp.data(), inp_size, inp_size);

            while ((int) draft.size()-1 < n_draft) {
                bool draft_success = false;
                for (int ngram_size = ngram_max; ngram_size >= ngram_min; --ngram_size) {
                    if (ngram_size > inp_size) {
                        continue;
                    }

                    all_token_hashmap & atc = all_token_counts[ngram_size - ngram_min];

                    const int ngram_start = inp_size-ngram_size + draft.size()-1;
                    uint64_t ngram = get_token(inp, draft, ngram_start);
                    for (int j = ngram_start; j < ngram_start + ngram_size; ++j) {
                        const uint64_t ngram_part = get_token(inp, draft, j);
                        ngram <<= 16;
                        ngram |= ngram_part;
                    }

                    all_token_hashmap::iterator token_counts_it = atc.find(ngram);
                    if (token_counts_it == atc.end()) {
                        continue;
                    }
                    const token_hashmap token_counts = token_counts_it->second;

                    int max_count = 0;
                    int sum_count = 0;
                    llama_token max_token = -1;

                    for (std::pair<llama_token, int> tc : token_counts) {
                        const llama_token token = tc.first;
                        const llama_token count = tc.second;

                        if (count > max_count) {
                            max_token = token;
                            max_count = count;
                        }
                        sum_count += count;
                    }
                    if (sum_count < draft_min_sample_size[ngram_size-1]) {
                        continue;
                    }
                    if (100*max_count < draft_min_percent[ngram_size-1]*sum_count) {
                        continue;
                    }

                    LOG(" - draft candidate: token=%d count=%d\n", max_token, max_count);
                    llama_batch_add(batch_tgt, max_token, n_past + draft.size(), { 0 }, true);
                    draft.push_back(max_token);
                    draft_success = true;
                    break;
                }

                if (!draft_success) {
                    break;
                }
            }
        };

        // draft already contains a single drafted token:
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        const int64_t t_start_draft_us = ggml_time_us();

        prompt_lookup();

        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;

        llama_decode(ctx, batch_tgt);
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("t_draft   = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx);

    llama_sampling_free(ctx_sampling);
    llama_batch_free(batch_tgt);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
