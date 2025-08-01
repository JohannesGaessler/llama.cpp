#!/usr/bin/env sh

export LLAMA_ARG_N_PARALLEL=1
export name=vllm

# for n_prompt in 2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576; do
for n_prompt in 8192 10240 12288 14336 16384 18432 20480 22528 24576; do
    for n_predict in 256 512 768 1024; do
        echo n_prompt=$n_prompt n_predict=$n_predict
        python3 server-bench.py \
            --path_server "http://0.0.0.0:${PORT}" \
            --name $name \
            --path_db server-bench.sqlite \
            --n_prompt $((32*$LLAMA_ARG_N_PARALLEL)) \
            --prompt_source rng-${n_prompt}-${n_prompt} \
            --n_predict_min $n_predict --n_predict $n_predict \
            --seed_offset -1
    done
done
