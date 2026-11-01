# Design note

Target: Tether, AI Inference Engineer (QVAC). The posting asks for a C++ systems layer for local inference: ggml and llama.cpp, GPU frameworks (CUDA, Vulkan, Metal, OpenCL), startup behaviour, memory pressure, throughput and latency balance, and long-session stability.

## Question

On one laptop, how do load time, memory, throughput and long-session behaviour of a small quantised language model change with threads, quantisation, load mode, GPU offload and context length?

## Method

- A C++17 program (`src/bench.cpp`) links `libllama` from llama.cpp and uses its public C API. It loads a model, prefills a synthetic prompt, decodes greedily, and prints one JSON line. It reads the resident set size and the peak resident set size from `/proc/self/status`.
- A Python runner (`scripts/run_matrix.py`) starts a fresh process for every run, repeats it three times, and keeps the median.
- Model: Qwen2.5-0.5B-Instruct in GGUF, Q4_K_M and Q8_0 (public, Apache 2.0).
- Matrix: thread sweep (1, 2, 4, 8, 12); Q4_K_M against Q8_0; load mode none, mmap and direct I/O; a long session of 3,000 generated tokens in a 4,096-token context with throughput recorded every 500 tokens.
- Two builds from the same source: CPU only, and CUDA for compute capability 8.6 with all layers offloaded.

## Scope limits chosen before the run

- One model size and one machine. Results do not transfer to other hardware.
- Page cache is warm: the files cannot be evicted without root. Load times are warm-start times.
- The prompt is synthetic token ids, so output text is meaningless. Only speed and memory are measured.
- No Vulkan, Metal or OpenCL build. No porting work on llama.cpp itself.
