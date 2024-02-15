#include "common.h"
#include "common/common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static void print_usage() {
    fprintf(stderr, "Merges multiple lookup cache files into a single one.\n");
    fprintf(stderr, "Usage: lookup-merge [--help] lookup_part_1.bin lookup_part_2.bin ... lookup_merged.bin\n");
}

int main(int argc, char ** argv){
    if (argc < 3) {
        print_usage();
        exit(1);
    }

    std::vector<std::string> args;
    args.resize(argc-1);
    for (int i = 0; i < argc-1; ++i) {
        args[i] = argv[i+1];
        if (args[i] == "-h" || args[i] == "--help") {
            print_usage();
            exit(0);
        }
    }

    llama_ngram_cache ngram_cache_merged = llama_ngram_cache_load(args[0]);

    for (size_t i = 1; i < args.size()-1; ++i) {
        llama_ngram_cache ngram_cache = llama_ngram_cache_load(args[i]);

        for (std::pair<uint64_t, llama_ngram_cache_part> ngram_part : ngram_cache) {
            const uint64_t         ngram = ngram_part.first;
            llama_ngram_cache_part  part = ngram_part.second;

            llama_ngram_cache::iterator part_merged_it = ngram_cache_merged.find(ngram);
            if (part_merged_it == ngram_cache_merged.end()) {
                ngram_cache_merged.emplace(ngram, part);
                continue;
            }

            for (std::pair<llama_token, int32_t> token_count : part) {
                const llama_token token = token_count.first;
                const int32_t     count = token_count.second;

                llama_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
                if (token_count_merged_it == part_merged_it->second.end()) {
                    part_merged_it->second.emplace(token, count);
                    continue;
                } else {
                    token_count_merged_it->second += count;
                }
            }
        }
    }

    // std::vector<llama_ngram_cache> ngram_cache(1);
    // llama_ngram_cache_update(ngram_cache, ngram_size, inp, inp.size(), true);
    // fprintf(stderr, "%s: hashing done, writing file\n", __func__);

    // llama_ngram_cache_save(ngram_cache, params.lookup_cache_static);
}
