#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
LLAMA_COMMIT=dde879bc4f954b809892fd0d660515f42536394a
uv venv .venv
uv pip install --python .venv/bin/python cmake ninja pytest
mkdir -p models third_party
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout "$LLAMA_COMMIT"
BASE=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
curl -L -o models/q4_k_m.gguf "$BASE/qwen2.5-0.5b-instruct-q4_k_m.gguf"
curl -L -o models/q8_0.gguf "$BASE/qwen2.5-0.5b-instruct-q8_0.gguf"
PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build-cpu -j 8
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j 6
