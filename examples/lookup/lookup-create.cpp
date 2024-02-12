#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

typedef std::unordered_map<llama_token, int32_t>        token_hashmap; // token -> number of times token has been seen
typedef std::unordered_map<uint64_t, token_hashmap> all_token_hashmap; // n-gram -> empirical distribution of following tokens
constexpr int ngram_size = 2;

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    GGML_ASSERT(model != nullptr);

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);

    const char * static_input_file = "./wikitext-2-raw/wiki.train.raw";
    std::ifstream file(static_input_file);
    if (!file) {
        fprintf(stderr, "error: failed to open file '%s'\n", static_input_file);
        exit(1);
    }
    std::string static_input;
    std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(static_input));
    if (!static_input.empty() && static_input.back() == '\n') {
        static_input.pop_back();
    }
    std::vector<llama_token> inp_static;
    inp_static = ::llama_tokenize(ctx, static_input, add_bos, true);
    fprintf(stderr, "lookup-create: tokenization done\n");

    auto update_hashmaps = [](all_token_hashmap * atc, const llama_token * inp_data, const int inp_size, const int nnew) -> void {
        // atcs = all_token_counts: the hashmaps to modify.
        // inp_data: the token sequence on which the hashmaps are based.
        // inp_size: the current size of inp_data.
        // nnew: how many new tokens have been appended to inp_data since the last call to this function.
        //
        // In order to get correct results inp_data can ONLY BE APPENDED TO.
        // Changes in the middle need a complete rebuild.

        const int     i_start    = std::max(inp_size - nnew, ngram_size);
        const int64_t t_start_ms = ggml_time_ms();
        int percentage_done = 0;
        for (int i = i_start; i < inp_size; ++i) {
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

            if (i >= inp_size*(percentage_done + 1)/100) {
                ++percentage_done;

                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (100 - percentage_done) * (t_now_ms - t_start_ms) / percentage_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - eta_min) / 1000;

                fprintf(stderr, "lookup-create: %02d%% done, ETA: %02ld:%02ld\n", percentage_done, eta_min, eta_s);
            }
        }
    };

    all_token_hashmap atc;
    update_hashmaps(&atc, inp_static.data(), inp_static.size(), inp_static.size());

    std::ofstream file_out("lookup.bin", std::ios::binary);
    for (std::pair<uint64_t, token_hashmap> item : atc) {
        const uint64_t ngram        = item.first;
        token_hashmap  token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();


        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(uint64_t));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }
}
