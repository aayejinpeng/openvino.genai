// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <cxxopts.hpp>
#include "read_prompt_from_file.h"
#include <fstream>
#include <typeinfo>

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("yjp,yjp_prefill_only_test", "test prefill for tokens/s", cxxopts::value<size_t>()->default_value(std::to_string(-1)))
    ("log_file", "Log file path", cxxopts::value<std::string>()->default_value("benchmark_log.csv"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string prompt;
    if (result.count("prompt") && result.count("prompt_file")) {
        std::cout << "Prompt and prompt file should not exist together!" << std::endl;
        return EXIT_FAILURE;
    } else {
        if (result.count("prompt_file")) {
            prompt = utils::read_prompt(result["prompt_file"].as<std::string>());
        } else {
            prompt = result["prompt"].as<std::string>().empty() ? "The Sky is blue because" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();
    config.apply_chat_template = false;

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();

    std::cout << ov::get_openvino_version() << std::endl;

    std::unique_ptr<ov::genai::LLMPipeline> pipe;

    if (result["yjp_prefill_only_test"].as<size_t>() != static_cast<size_t>(-1)) {
        config.max_new_tokens = 1;
        config.max_length = result["yjp_prefill_only_test"].as<size_t>()+1;
        // config.
        size_t prefill_only_test_tokens = result["yjp_prefill_only_test"].as<size_t>();

        prompt = "oooo"; // dummy prompt to generate required number of tokens
        std::cout << "LLM pipe is being created..." << std::endl;
        pipe = std::make_unique<ov::genai::LLMPipeline>(
            models_path, device, 
            // ov::genai::scheduler_config(scheduler_config),
            std::pair<std::string, ov::Any>{"ATTENTION_BACKEND", std::string("SDPA")},
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::enable_profiling(true)
        );
        std::cout << "Backend: " << pipe->backend_name() << std::endl;

        std::cout << "Backend type: " << typeid(*pipe).name() << std::endl;

        auto input_data = pipe->get_tokenizer().encode(prompt,ov::genai::add_special_tokens(true),ov::genai::pad_to_max_length(true), ov::genai::max_length(result["yjp_prefill_only_test"].as<size_t>()));
        size_t prompt_token_size = input_data.input_ids.get_shape()[1];
        std::cout << "Prompt token size:" << prompt_token_size << std::endl;

        for (size_t i = 0; i < num_warmup; i++)
            pipe->generate(input_data, config);

        auto res = pipe->generate(input_data, config);
        ov::genai::PerfMetrics metrics = res.perf_metrics;

        prompt = "YJP is the best bug killer";

        while (prompt.size() < prefill_only_test_tokens * 10)
        {
            prompt += " YJP is the best bug killer" + prompt;
        }
        for (size_t i = 0; i < num_iter - 1; i++) {
            input_data = pipe->get_tokenizer().encode(prompt,ov::genai::pad_to_max_length(true), ov::genai::max_length(result["yjp_prefill_only_test"].as<size_t>()));
            res = pipe->generate(input_data, config);
            metrics = res.perf_metrics;
        }

        std::cout << "Prefill only test, input tokens: " << metrics.get_num_input_tokens() << std::endl;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Output token size:" << res.perf_metrics.get_num_generated_tokens() << std::endl;
        std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
        std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
        std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
        std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
        std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
        std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
        std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

                // Print per-layer profiling info if available
        auto* infer_req = pipe->get_infer_request();
        if (infer_req) {
            try {
            auto prof_info = infer_req->get_profiling_info();
            if (!prof_info.empty()) {
                std::ofstream log_file(result["log_file"].as<std::string>());
                log_file << "\n=== Per-layer Profiling Info ===" << std::endl;
                log_file << std::left << std::setw(100) << "Layer Name" 
                      << std::right << std::setw(15) << "Time (ms)" << std::endl;
                log_file << std::string(115, '-') << std::endl;
                double total_time = 0.0;
                for (const auto& layer : prof_info) {
                // layer.real_time is in microseconds as duration, convert to milliseconds
                double time_ms = layer.real_time.count() / 1000.0;
                total_time += time_ms;
                log_file << std::left << std::setw(100) << layer.node_name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                      << std::endl;
                }
                log_file << std::string(115, '-') << std::endl;
                log_file << std::left << std::setw(100) << "TOTAL"
                      << std::right << std::setw(15) << std::fixed << std::setprecision(3) << total_time
                      << std::endl;
                log_file.close();
            }
            } catch (const std::exception& e) {
            std::cout << "Note: Profiling info not available: " << e.what() << std::endl;
            }
        }

    }
    else
    {
        if (device == "NPU")
            pipe = std::make_unique<ov::genai::LLMPipeline>(
                models_path, device, 
                ov::enable_profiling(true)
            );
        else
            pipe = std::make_unique<ov::genai::LLMPipeline>(
                models_path, device, 
                ov::genai::scheduler_config(scheduler_config),
                ov::enable_profiling(true)
            );
        std::cout << "Backend: " << pipe->backend_name() << std::endl;

        auto input_data = pipe->get_tokenizer().encode(prompt);
        size_t prompt_token_size = input_data.input_ids.get_shape()[1];
        std::cout << "Prompt token size:" << prompt_token_size << std::endl;

        for (size_t i = 0; i < num_warmup; i++)
            pipe->generate(prompt, config);

        ov::genai::DecodedResults res = pipe->generate(prompt, config);
        ov::genai::PerfMetrics metrics = res.perf_metrics;
        for (size_t i = 0; i < num_iter - 1; i++) {
            res = pipe->generate(prompt, config);
            metrics = metrics + res.perf_metrics;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Output token size:" << res.perf_metrics.get_num_generated_tokens() << std::endl;
        std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
        std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
        std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
        std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
        std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
        std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
        std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

        // Print per-layer profiling info if available
        auto* infer_req = pipe->get_infer_request();
        if (infer_req) {
            try {
                auto prof_info = infer_req->get_profiling_info();
                if (!prof_info.empty()) {
                    std::cout << "\n=== Per-layer Profiling Info ===" << std::endl;
                    std::cout << std::left << std::setw(60) << "Layer Name" 
                              << std::right << std::setw(15) << "Time (ms)" << std::endl;
                    std::cout << std::string(75, '-') << std::endl;
                    double total_time = 0.0;
                    for (const auto& layer : prof_info) {
                        // layer.real_time is in microseconds as duration, convert to milliseconds
                        double time_ms = layer.real_time.count() / 1000.0;
                        total_time += time_ms;
                        std::cout << std::left << std::setw(60) << layer.node_name
                                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                                  << std::endl;
                    }
                    std::cout << std::string(75, '-') << std::endl;
                    std::cout << std::left << std::setw(60) << "TOTAL"
                              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << total_time
                              << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "Note: Profiling info not available: " << e.what() << std::endl;
            }
        }
    }
    

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
