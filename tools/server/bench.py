#!/usr/bin/env python3

import argparse
import json
import subprocess
from time import sleep, time

import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.contrib.concurrent import thread_map


def get_prompts(n_prompts: int) -> list[str]:
    ret = datasets.load_dataset("cais/mmlu", "all")["test"]["question"]
    if n_prompts >= 0:
        ret = ret[:n_prompts]
    return ret


TEMPLATE_SERVER_ADDRESS = "http://localhost:{port}"


def get_server(path_server: str, path_model: str, port: int, parallel: int, ctx_size: int) -> dict:
    address = TEMPLATE_SERVER_ADDRESS.format(port=port)

    popen_args: list[str] = [
        path_server,
        "--flash-attn",
        "--n-gpu-layers", "999",
        "--parallel", str(parallel),
        "--ctx-size", str(parallel * ctx_size),
        "--model", path_model,
        "--port", str(port),
        "--swa-full",  # FIXME
        "--attn-streams",
    ]
    fout = open("bench.log", "w")
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


def benchmark(path_server: str, path_model: str, port: int, parallel: int, ctx_size: int, n_prompts: int, n_predict: int):
    prompts: list[str] = get_prompts(n_prompts)

    server = None
    try:
        server: dict = get_server(path_server, path_model, port, parallel, ctx_size)
        server_address: str = server["address"]

        with requests.Session() as session:
            data: list[dict] = []
            for p in prompts:
                data.append({"session": session, "server_address": server_address, "prompt": p, "n_predict": n_predict})

            t0 = time()
            results: list[tuple[int, int, list[float]]] = thread_map(send_prompt, data, max_workers=parallel + 1, chunksize=1)
    finally:
        if server is not None:
            server["process"].terminate()
            server["fout"].close()
            server["process"].wait()

    x = []
    y = []
    for (prompt_n, prompt_ms, _) in results:
        x.append(prompt_n)
        y.append(prompt_ms)
    x = np.array(x, dtype=np.int64)
    y = np.array(y, dtype=np.float64)

    plt.figure()
    plt.scatter(x, y, marker=",")
    plt.xlim(0, 1.05 * np.max(x))
    plt.ylim(0, 1.05 * np.max(y))
    plt.xlabel("Prompt length")
    plt.ylabel("Time to first token [ms]")
    plt.savefig("prompt_time.png", dpi=240)

    x = []
    for (_, _, token_arrival_times) in results:
        x += token_arrival_times
    x = np.array(x, dtype=np.float64)
    x -= t0

    plt.figure()
    plt.hist(x, np.arange(0, np.ceil(np.max(x)) + 1))
    plt.xlim(0, np.ceil(np.max(x)) + 2)
    plt.xlabel("Time [s]")
    plt.ylabel("Num. tokens generated per second")
    plt.savefig("gen_time.png", dpi=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_server", type=str, default="llama-server")
    parser.add_argument("--path_model", type=str, required=True)
    parser.add_argument("--port", type=int, default=18725)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--ctx_size", type=int, default=4096)
    parser.add_argument("--n_prompts", type=int, default=250)
    parser.add_argument("--n_predict", type=int, default=2048)
    args = parser.parse_args()
    benchmark(args.path_server, args.path_model, args.port, args.parallel, args.ctx_size, args.n_prompts, args.n_predict)
