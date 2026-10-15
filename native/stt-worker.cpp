// One model context per process; framed PCM requests over private pipes.
// Protocol v1: uint32 LE byte count, followed by 16 kHz mono s16le PCM.
// Replies use the same length prefix and contain UTF-8 text. Zero input exits.
#include "whisper.h"
#include "parakeet.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static bool read_size(uint32_t & size) {
    std::array<unsigned char, 4> bytes{};
    std::cin.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    if (std::cin.gcount() == 0) return false;
    if (std::cin.gcount() != 4) throw std::runtime_error("truncated frame header");
    size = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
           uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
    return true;
}

static void reply(const std::string & text) {
    uint32_t size = static_cast<uint32_t>(text.size());
    const unsigned char bytes[] = {
        static_cast<unsigned char>(size), static_cast<unsigned char>(size >> 8),
        static_cast<unsigned char>(size >> 16), static_cast<unsigned char>(size >> 24),
    };
    std::cout.write(reinterpret_cast<const char *>(bytes), 4);
    std::cout.write(text.data(), text.size());
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("response pipe closed");
}

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: kdictate-stt-worker whisper|parakeet MODEL [THREADS] [LANGUAGE] [BEAM] [NO_SPEECH] [cpu]");
        const std::string engine = argv[1];
        if (engine != "whisper" && engine != "parakeet") throw std::runtime_error("unknown engine");
        const int threads = argc > 3 ? std::stoi(argv[3]) : 6;
        const std::string language = argc > 4 ? argv[4] : "en";
        const int beam = argc > 5 ? std::stoi(argv[5]) : 3;
        const float no_speech = argc > 6 ? std::stof(argv[6]) : 0.6f;
        const bool gpu = !(argc > 7 && std::string(argv[7]) == "cpu");
        if (threads < 1 || beam < 1) throw std::runtime_error("invalid thread or beam count");
        ggml_backend_load_all();
        if (gpu) {
            bool found = false;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
                found |= type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
            }
            if (!found) throw std::runtime_error("GPU requested but no GPU backend is available");
        }
        std::unique_ptr<whisper_context, decltype(&whisper_free)> wc(nullptr, whisper_free);
        std::unique_ptr<parakeet_context, decltype(&parakeet_free)> pc(nullptr, parakeet_free);
        if (engine == "whisper") {
            auto params = whisper_context_default_params();
            params.use_gpu = gpu;
            params.flash_attn = true;
            wc.reset(whisper_init_from_file_with_params(argv[2], params));
            if (!wc) throw std::runtime_error("Whisper model initialization failed");
        } else {
            auto params = parakeet_context_default_params();
            params.use_gpu = gpu;
            pc.reset(parakeet_init_from_file_with_params(argv[2], params));
            if (!pc) throw std::runtime_error("Parakeet model initialization failed");
        }
        reply("READY 1 " + engine);
        uint32_t size;
        while (read_size(size) && size != 0) {
            if (size % 2 || size > 16000 * 2 * 60) throw std::runtime_error("invalid PCM frame size (maximum 60 seconds)");
            std::vector<unsigned char> bytes(size);
            std::cin.read(reinterpret_cast<char *>(bytes.data()), size);
            if (std::cin.gcount() != size) throw std::runtime_error("truncated PCM frame");
            std::vector<float> pcm(size / 2);
            for (size_t i = 0; i < pcm.size(); ++i) {
                const auto sample = static_cast<int16_t>(uint16_t(bytes[2*i]) | uint16_t(bytes[2*i+1]) << 8);
                pcm[i] = sample / 32768.0f;
            }
            std::string text;
            if (wc) {
                // Match whisper-cli: retain greedy best_of=5 for temperature
                // fallback, then select beam search for the initial decode.
                auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                params.strategy = beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
                params.n_threads = threads;
                params.language = language.c_str();
                params.beam_search.beam_size = beam;
                params.no_speech_thold = no_speech;
                params.no_context = true;
                params.no_timestamps = true;
                params.print_special = params.print_progress = params.print_realtime = params.print_timestamps = false;
                if (whisper_full(wc.get(), params, pcm.data(), pcm.size())) throw std::runtime_error("Whisper decode failed");
                for (int i = 0; i < whisper_full_n_segments(wc.get()); ++i) text += whisper_full_get_segment_text(wc.get(), i);
            } else {
                auto params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
                params.n_threads = threads;
                params.no_context = true;
                if (parakeet_full(pc.get(), params, pcm.data(), pcm.size())) throw std::runtime_error("Parakeet decode failed");
                for (int i = 0; i < parakeet_full_n_segments(pc.get()); ++i) text += parakeet_full_get_segment_text(pc.get(), i);
            }
            reply(text);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "kdictate-stt-worker: " << error.what() << '\n';
        return 1;
    }
}
