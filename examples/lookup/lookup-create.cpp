#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int ngram_min = 1;
constexpr int ngram_max = 4;

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    GGML_ASSERT(model != nullptr);

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
    fprintf(stderr, "%s: tokenization done\n", __func__);


    llama_ngram_cache ngram_cache;
    llama_ngram_cache_update(ngram_cache, ngram_min, ngram_max, inp, inp.size(), true);
    fprintf(stderr, "%s: hashing done, writing file\n", __func__);

    llama_ngram_cache_save(ngram_cache, params.lookup_cache_static);
}
