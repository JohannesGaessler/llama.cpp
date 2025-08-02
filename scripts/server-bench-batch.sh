#!/usr/bin/env sh

export name=vllm

for n_parallel in 1 2 4 8 16; do
    export LLAMA_ARG_N_PARALLEL=$n_parallel
    # for n_prompt in 4096 6144 10240 12288 14336 18432 20480 22528; do
    for n_prompt in 2048 24576 8192 16384; do
        for n_predict in 256 512 768 1024; do
            echo n_parallel=$n_parallel n_prompt=$n_prompt n_predict=$n_predict
            python3 server-bench.py \
                --path_server "http://0.0.0.0:${PORT}" \
                --name $name \
                --path_db server-bench.sqlite \
                --n_prompt $((32 * $n_parallel)) \
                --prompt_source rng-${n_prompt}-${n_prompt} \
                --n_predict_min $n_predict --n_predict $n_predict \
                --seed_offset -1
        done
    done
done
