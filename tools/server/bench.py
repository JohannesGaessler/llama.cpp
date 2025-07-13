#!/usr/bin/env python3

import argparse
import json
import subprocess
from time import sleep, time
from typing import Optional

import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.contrib.concurrent import thread_map


def get_prompts(n_prompts: int) -> list[str]:
    print("Loading MMLU dataset...")
    ret = datasets.load_dataset("cais/mmlu", "all")["test"]["question"]
    if n_prompts >= 0:
        ret = ret[:n_prompts]
    return ret


TEMPLATE_SERVER_ADDRESS = "http://localhost:{port}"


def get_server(path_server: str, path_model: str, path_log: Optional[str], port: int, n_gpu_layers: int, parallel: int, ctx_size: int) -> dict:
    print("Starting the llama.cpp server...")
    address = TEMPLATE_SERVER_ADDRESS.format(port=port)

    popen_args: list[str] = [
        path_server,
        "--flash-attn",
        "--n-gpu-layers", str(n_gpu_layers),
        "--parallel", str(parallel),
        "--ctx-size", str(parallel * ctx_size),
        "--model", path_model,
        "--port", str(port),
        "--swa-full",  # FIXME performance bad otherwise
        "--attn-streams",
    ]
    fout = open("bench.log", "w") if path_log is not None else subprocess.DEVNULL
    process = subprocess.Popen(popen_args, stdout=fout, stderr=subprocess.STDOUT)

    n_failures: int = 0
    while True:
        try:
            sleep(1.0)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"llama.cpp server for {path_model} exited unexpectedly with exit code {exit_code}")
            response = requests.get(f"{address}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            n_failures += 1
            if n_failures >= 10:
                raise RuntimeError(f"llama.cpp server for {path_model} is not healthy after 10 seconds")

    return dict(process=process, address=address, fout=fout)


def send_prompt(data: dict) -> tuple[int, float, list[float]]:
    session = data["session"]
    server_address: str = data["server_address"]

    response = session.post(
        f"{server_address}/apply-template",
        json={"messages": [{"role": "user", "content": data["prompt"], "stream": True}]}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]

    json_data: dict = {"prompt": prompt, "n_predict": data["n_predict"], "stream": True}
    response = session.post(f"{server_address}/completion", json=json_data, stream=True)

    token_arrival_times: list[float] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line.startswith("data: "):
            continue
        last_valid_line = line
        token_arrival_times.append(time())
    token_arrival_times = token_arrival_times[:-1]

    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    timings: dict = json.loads(last_valid_line[6:])["timings"]

    return (timings["prompt_n"], timings["prompt_ms"], token_arrival_times)


def benchmark(path_server: str, path_model: str, path_log: Optional[str], port: int, n_gpu_layers: int, parallel: int, ctx_size: int, n_prompts: int, n_predict: int):
    prompts: list[str] = get_prompts(n_prompts)

    server = None
    try:
        server: dict = get_server(path_server, path_model, path_log, port, n_gpu_layers, parallel, ctx_size)
        server_address: str = server["address"]

        print("Starting the benchmark...")
        print()
        with requests.Session() as session:
            data: list[dict] = []
            for p in prompts:
                data.append({"session": session, "server_address": server_address, "prompt": p, "n_predict": n_predict})

            t0 = time()
            results: list[tuple[int, int, list[float]]] = thread_map(send_prompt, data, max_workers=parallel + 1, chunksize=1)
    finally:
        if server is not None:
            server["process"].terminate()
            if path_log is not None:
                server["fout"].close()
            server["process"].wait()

    x = []
    y = []
    for (prompt_n, prompt_ms, _) in results:
        x.append(prompt_n)
        y.append(prompt_ms)
    x = np.array(x, dtype=np.int64)
    y = np.array(y, dtype=np.float64)

    print()
    print(f"Average prompt length:             {np.mean(x):.2f} tokens")
    print(f"Average prompt latency:            {np.mean(y):.2f} ms")
    print(f"Average prompt speed:              {np.sum(x) / (1e-3 * np.sum(y)):.2f} tokens/s")

    plt.figure()
    plt.scatter(x, y, s=10.0, marker=".", alpha=0.25)
    plt.xlim(0, 1.05 * np.max(x))
    plt.ylim(0, 1.05 * np.max(y))
    plt.title(path_model)
    plt.xlabel("Prompt length [tokens]")
    plt.ylabel("Time to first token [ms]")
    plt.savefig("prompt_time.png", dpi=240)

    depth_sum: int = 0
    x = []
    for (prompt_n, _, token_arrival_times) in results:
        n_tokens: int = len(token_arrival_times)
        depth_sum += n_tokens * prompt_n
        depth_sum += n_tokens * (n_tokens + 1) // 2
        x += token_arrival_times
    x = np.array(x, dtype=np.float64)
    x -= t0
    x_max = np.max(x)

    print(f"Average generation depth:          {depth_sum / x.shape[0]:.2f} tokens")
    print(f"Average total generation speed:    {x.shape[0] / x_max:.2f} tokens/s = {x.shape[0]} tokens / {x_max:.2f} s")
    print(f"Average generation speed per slot: {x.shape[0] / (parallel * x_max):.2f} tokens/s / slot")

    x_bin_max = np.ceil(x_max) + 1
    plt.figure()
    plt.hist(x, np.arange(0, x_bin_max))
    plt.xlim(0, x_bin_max + 1)
    plt.title(path_model)
    plt.xlabel("Time [s]")
    plt.ylabel("Num. tokens generated per second")
    plt.savefig("gen_rate.png", dpi=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_server", type=str, default="llama-server", help="Path to the llama.cpp server binary")
    parser.add_argument("--path_model", type=str, required=True, help="Path to the model to use for the benchmark")
    parser.add_argument("--path_log", type=str, default=None, help="Path to the model to use for the benchmark")
    parser.add_argument("--port", type=int, default=18725, help="Port to use for the server during the benchmark")
    parser.add_argument("--n_gpu_layers", type=int, default=999, help="Number of GPU layers for the server")
    parser.add_argument("--parallel", type=int, default=16, help="Number of slots for the server")
    parser.add_argument("--ctx_size", type=int, default=4096, help="Server context size per slot")
    parser.add_argument("--n_prompts", type=int, default=250, help="Number of prompts to evaluate")
    parser.add_argument("--n_predict", type=int, default=2048, help="Max. number of tokens to predict per prompt")
    args = parser.parse_args()
    benchmark(**vars(args))
