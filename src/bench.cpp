#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

struct Options {
    std::string model_path;
    int threads = 4;
    int prompt_tokens = 512;
    int generate_tokens = 128;
    int context = 2048;
    int gpu_layers = 0;
    int load_mode = LLAMA_LOAD_MODE_MMAP;
    int report_every = 0;
};

struct Phase {
    double seconds = 0.0;
    int tokens = 0;
    double rate() const { return seconds > 0.0 ? tokens / seconds : 0.0; }
};

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

long status_kilobytes(const std::string& key) {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(key, 0) == 0) {
            return std::strtol(line.c_str() + key.size() + 1, nullptr, 10);
        }
    }
    return -1;
}

Options parse(int argc, char** argv) {
    Options options;
    options.model_path = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const int value = std::atoi(argv[i + 1]);
        if (key == "--threads") options.threads = value;
        else if (key == "--prompt") options.prompt_tokens = value;
        else if (key == "--generate") options.generate_tokens = value;
        else if (key == "--context") options.context = value;
        else if (key == "--gpu-layers") options.gpu_layers = value;
        else if (key == "--load-mode") options.load_mode = value;
        else if (key == "--report-every") options.report_every = value;
    }
    return options;
}

std::vector<llama_token> synthetic_prompt(const llama_vocab* vocab, int count) {
    std::vector<llama_token> tokens;
    const int span = 1000;
    tokens.push_back(llama_vocab_bos(vocab) >= 0 ? llama_vocab_bos(vocab) : 1);
    for (int i = 1; i < count; ++i) tokens.push_back(1000 + (i * 7919) % span);
    return tokens;
}

Phase prefill(llama_context* context, std::vector<llama_token>& tokens) {
    const auto start = Clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(context, batch) != 0) std::exit(2);
    return {elapsed(start), static_cast<int>(tokens.size())};
}

Phase decode(llama_context* context, llama_sampler* sampler, int count, std::vector<double>& checkpoints, int report_every) {
    const auto start = Clock::now();
    auto window_start = start;
    int in_window = 0;
    for (int i = 0; i < count; ++i) {
        llama_token token = llama_sampler_sample(sampler, context, -1);
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(context, batch) != 0) std::exit(3);
        ++in_window;
        if (report_every > 0 && in_window == report_every) {
            checkpoints.push_back(in_window / elapsed(window_start));
            window_start = Clock::now();
            in_window = 0;
        }
    }
    return {elapsed(start), count};
}

void print_checkpoints(const std::vector<double>& rates) {
    std::printf(",\"window_rates\":[");
    for (size_t i = 0; i < rates.size(); ++i) std::printf("%s%.2f", i ? "," : "", rates[i]);
    std::printf("]");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    const Options options = parse(argc, argv);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    model_params.load_mode = static_cast<llama_load_mode>(options.load_mode);
    const auto load_start = Clock::now();
    llama_model* model = llama_model_load_from_file(options.model_path.c_str(), model_params);
    if (!model) return 4;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = options.context;
    context_params.n_batch = options.prompt_tokens;
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads;
    llama_context* context = llama_init_from_model(model, context_params);
    if (!context) return 5;
    const double load_seconds = elapsed(load_start);
    const long rss_after_load = status_kilobytes("VmRSS");
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    std::vector<llama_token> prompt = synthetic_prompt(vocab, options.prompt_tokens);
    const Phase prompt_phase = prefill(context, prompt);
    std::vector<double> rates;
    const Phase generate_phase = decode(context, sampler, options.generate_tokens, rates, options.report_every);
    std::printf("{\"load_seconds\":%.3f,\"rss_kb_after_load\":%ld,\"rss_kb_peak\":%ld,\"prefill_tokens_per_second\":%.2f,"
                "\"decode_tokens_per_second\":%.2f,\"threads\":%d,\"gpu_layers\":%d,\"load_mode\":%d,\"model_bytes\":%llu",
                load_seconds, rss_after_load, status_kilobytes("VmHWM"), prompt_phase.rate(), generate_phase.rate(),
                options.threads, options.gpu_layers, options.load_mode,
                static_cast<unsigned long long>(llama_model_size(model)));
    print_checkpoints(rates);
    std::printf("}\n");
    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
