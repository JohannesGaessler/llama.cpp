#!/usr/bin/env sh

export LLAMA_ARG_N_PARALLEL=1
export name=vllm

for n_prompt in 2048 4096 8196 12288 16376 24576; do
    for n_predict in 256 512 768 1024; do
        echo n_prompt=$n_prompt n_predict=$n_predict
        echo python3 server-bench.py \
            --path_server "http://0.0.0.0:${PORT}" \
            --name $name \
            --path_db server-bench.sqlite \
            --n_prompt $((32*$LLAMA_ARG_N_PARALLEL)) \
            --prompt_source rng-${n_prompt}-${n_prompt} \
            --n_predict_min $n_predict --n_predict $n_predict
    done
done
