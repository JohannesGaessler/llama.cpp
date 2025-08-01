#!/usr/bin/env sh

export LLAMA_ARG_N_PARALLEL=1

for n_prompt in 2048 4096 8196 12288 16376 24576; do
    for n_predict in 256 512 768 1024; do
        echo n_prompt=$n_prompt n_predict=$n_predict
    done
done
