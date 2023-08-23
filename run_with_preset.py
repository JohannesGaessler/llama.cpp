#!/usr/bin/env python3

import os
import subprocess
import sys

import yaml

CLI_ARGS_MAIN = [
    "batch-size", "cfg-negative-prompt", "cfg-scale", "chunks", "color", "ctx-size", "escape",
    "export", "file", "frequency-penalty", "grammar", "grammar-file", "hellaswag",
    "hellaswag-tasks", "ignore-eos", "in-prefix", "in-prefix-bos", "in-suffix", "instruct",
    "interactive", "interactive-first", "keep", "logdir", "logit-bias", "lora", "lora-base",
    "low-vram", "main-gpu", "memory-f32", "mirostat", "mirostat-ent", "mirostat-lr", "mlock",
    "model", "mtest", "mul-mat-q", "multiline-input", "n-gpu-layers", "n-predict", "n_probs",
    "no-mmap", "np-penalize-nl", "numa", "perplexity", "ppl_output_type", "ppl_stride",
    "presence-penalty", "prompt", "prompt-cache", "prompt-cache-all", "prompt-cache-ro",
    "random-prompt", "repeat-last-n", "repeat-penalty", "reverse-prompt", "rope-freq-base",
    "rope-freq-scale", "rope-scale", "seed", "simple-io", "tensor-split", "threads",
    "temp", "tfs", "top-k", "top-p", "typical", "verbose-prompt"
]

with open(sys.argv[1], "r") as f:
    props = yaml.load(f, yaml.SafeLoader)

props = {prop.replace("_", "-"): val for prop, val in props.items()}

binary = props.pop("binary", "main")
if os.path.exists(f"./{binary}"):
    binary = f"./{binary}"

command_list = [binary]

for cli_arg in CLI_ARGS_MAIN:
    value = props.get(cli_arg, None)

    if not value or value == -1:
        continue

    if cli_arg == "logit-bias":
        for token, bias in value.items():
            command_list.append("--logit-bias")
            command_list.append(f"{token}{bias}")
        continue

    command_list.append(f"--{cli_arg}")

    if cli_arg == "tensor-split":
        command_list.append(",".join([str(v) for v in value]))
        continue

    value = str(value)

    if value != "True":
        command_list.append(str(value))

print(command_list)

result = subprocess.run(command_list)
sys.exit(result.returncode)
