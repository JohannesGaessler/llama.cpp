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

    std::vector<llama_ngram_cache> ngram_cache_merged;
    ngram_cache_merged.push_back(llama_ngram_cache_load(args[0]));

    for (size_t i = 1; i < args.size()-1; ++i) {
        fprintf(stderr, "lookup-merge: loading file %s\n", args[i].c_str());
        llama_ngram_cache ngram_cache = llama_ngram_cache_load(args[i]);

        for (std::pair<uint64_t, llama_ngram_cache_part> ngram_part : ngram_cache) {
            const uint64_t         ngram = ngram_part.first;
            llama_ngram_cache_part  part = ngram_part.second;

            llama_ngram_cache::iterator part_merged_it = ngram_cache_merged[0].find(ngram);
            if (part_merged_it == ngram_cache_merged[0].end()) {
                ngram_cache_merged[0].emplace(ngram, part);
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

    fprintf(stderr, "lookup-merge: saving file %s\n", args.back().c_str());
    llama_ngram_cache_save(ngram_cache_merged, args.back());
}
