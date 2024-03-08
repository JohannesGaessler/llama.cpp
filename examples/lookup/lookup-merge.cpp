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
        fprintf(stderr, "lookup-merge: loading file %s\n", args[i].c_str());
        llama_ngram_cache ngram_cache = llama_ngram_cache_load(args[i]);

        llama_ngram_cache_merge(ngram_cache_merged, ngram_cache);
    }

    fprintf(stderr, "lookup-merge: saving file %s\n", args.back().c_str());
    llama_ngram_cache_save(ngram_cache_merged, args.back());
}
