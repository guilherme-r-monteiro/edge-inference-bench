# edge-inference-bench

A C++ benchmark harness over llama.cpp that measures load time, memory, throughput and long-session behaviour of a small quantised language model on one laptop, on the CPU and on the GPU.

Design note: `docs/design.md`. Raw results: `results/cpu-matrix.json`, `results/cuda-matrix.json`.

- Harness: `src/bench.cpp` (C++17) links `libllama` and uses its public C API. It reads RSS and peak RSS from `/proc/self/status`.
- Runner: `scripts/run_matrix.py` starts a fresh process per run, repeats three times and keeps the median.
- Model: Qwen2.5-0.5B-Instruct, GGUF Q4_K_M (485 MB) and Q8_0 (676 MB).
- Machine: 12 CPU threads, NVIDIA RTX 3050 6 GB laptop GPU, 14 GiB RAM.
- llama.cpp commit `dde879bc4f954b809892fd0d660515f42536394a`.

## Results

Prompt of 512 tokens and 128 generated tokens, greedy, median of three runs. Tokens per second.

Thread sweep, Q4_K_M, CPU:

| Threads | Prefill | Decode |
|---|---|---|
| 1 | 102 | 31.8 |
| 2 | 194 | 51.1 |
| 4 | 315 | 75.6 |
| 8 | 308 | 76.6 |
| 12 | 376 | 67.3 |

Decode stops scaling at 4 threads and drops at 12. Prefill keeps scaling.

Quantisation and device, 4 threads:

| Device | Quantisation | Prefill | Decode | Peak RSS |
|---|---|---|---|---|
| CPU | Q4_K_M | 330 | 78.0 | 613 MB |
| CPU | Q8_0 | 391 | 65.9 | 759 MB |
| GPU, all layers | Q4_K_M | 8,997 | 265.6 | 727 MB |
| GPU, all layers | Q8_0 | 10,905 | 233.9 | 903 MB |

Q8_0 decodes 15% slower than Q4_K_M on the CPU and 12% slower on the GPU, and it prefills faster on both. Decode is limited by memory bandwidth, so the smaller file wins. Prefill is limited by compute, and the Q8_0 kernels do less dequantisation work.

Load mode, Q8_0, warm page cache:

| Mode | Load seconds | Peak RSS (CPU) | Peak RSS (GPU) |
|---|---|---|---|
| none (read into memory) | 0.341 | 726 MB | 457 MB |
| mmap | 0.116 | 726 MB | 903 MB |
| direct I/O | 0.511 | 728 MB | 652 MB |

On the CPU, mmap loads three times faster than a plain read. With GPU offload, mmap gives the highest peak RSS because mapped file pages stay resident while weights are copied to the device. A plain read has the lowest peak (457 MB).

Long session, Q4_K_M, 3,000 generated tokens, context 4,096. Decode tokens per second in windows of 500 tokens:

| Device | Window 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CPU | 73.1 | 70.6 | 60.5 | 36.0 | 34.1 | 34.0 |
| GPU | 264.3 | 260.5 | 258.2 | 254.7 | 250.1 | 246.2 |

On the CPU decode speed halves as the context grows from about 500 to 3,500 tokens. On the GPU it falls by 7%. Peak RSS grew from 595 MB after load to 637 MB on the CPU and stayed at 727 MB on the GPU, with no sign of a leak over 3,000 tokens.

## Limits

- One model (0.5 B parameters) and one machine. A larger model changes every ratio above.
- Warm page cache. Cold-start load times are not measured.
- Only three runs per setting. The thread counts 4 and 8 are within noise of each other for decode.
- Synthetic prompt tokens. The harness measures speed and memory, not output quality. No perplexity or accuracy check was run on the two quantisations.
- The GPU rows with `load_*` labels use a 64-token prompt, so their prefill speed is lower than the 512-token rows.
- CUDA only. No Vulkan, Metal or OpenCL build. No changes to llama.cpp itself were made, so this shows measurement skill on top of it, not porting or kernel work.
- Windows of 500 tokens average out short stalls.

## Run it

```sh
scripts/setup.sh
.venv/bin/python -m pytest
python3 scripts/run_matrix.py build-cpu/bench results/cpu-matrix.json
python3 scripts/run_matrix.py build-cuda/bench results/cuda-matrix.json 99
```
