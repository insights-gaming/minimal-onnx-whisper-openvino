#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <complex>
#include <sndfile.h>
#include <fftw3.h>
#include <onnxruntime_cxx_api.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Base64 decoding function
std::string base64_decode(const std::string& encoded_string) {
    const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string decoded;
    int val = 0, valb = -8;
    for (unsigned char c : encoded_string) {
        if (c == '=') break;
        if (chars.find(c) == std::string::npos) continue;
        val = (val << 6) + chars.find(c);
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

// Load tokens from file and decode base64
std::unordered_map<int32_t, std::string> load_tokens(const std::string& filename) {
    std::unordered_map<int32_t, std::string> tokens;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string base64_token = line.substr(0, space_pos);
            int32_t token_id = std::stoi(line.substr(space_pos + 1));
            std::string decoded_token = base64_decode(base64_token);
            tokens[token_id] = decoded_token;
        }
    }
    
    return tokens;
}

// Load audio file
std::vector<float> load_audio(const std::string& filename, int& sample_rate) {
    SF_INFO info;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);
    if (!file) {
        throw std::runtime_error("Failed to open audio file: " + filename);
    }
    
    sample_rate = info.samplerate;
    std::vector<float> audio_data(info.frames);
    sf_readf_float(file, audio_data.data(), info.frames);
    sf_close(file);
    
    return audio_data;
}

// Apply Hann window
void apply_hann_window(std::vector<float>& frame) {
    int n = frame.size();
    for (int i = 0; i < n; i++) {
        frame[i] *= 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
    }
}

// Convert to mel scale
double hz_to_mel(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Create mel filter bank (improved version)
std::vector<std::vector<float>> create_mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    int n_freqs = n_fft / 2 + 1;
    
    // Create mel points using proper whisper mel scale
    double mel_low = hz_to_mel(0);
    double mel_high = hz_to_mel(sample_rate / 2.0);
    std::vector<double> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }
    
    // Convert back to Hz
    std::vector<double> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert to FFT bin numbers
    std::vector<double> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        bin_points[i] = (n_fft + 1) * hz_points[i] / sample_rate;
    }
    
    // Create triangular filter bank
    std::vector<std::vector<float>> filters(n_mels, std::vector<float>(n_freqs, 0.0f));
    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_freqs; k++) {
            double freq = (double)k;
            if (freq >= bin_points[m] && freq <= bin_points[m + 1]) {
                filters[m][k] = (freq - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
            } else if (freq >= bin_points[m + 1] && freq <= bin_points[m + 2]) {
                filters[m][k] = (bin_points[m + 2] - freq) / (bin_points[m + 2] - bin_points[m + 1]);
            }
        }
        
        // Normalize filter
        float sum = 0.0f;
        for (float f : filters[m]) sum += f;
        if (sum > 0) {
            for (float& f : filters[m]) f /= sum;
        }
    }
    
    return filters;
}

// Extract mel spectrogram features (improved version)
std::vector<std::vector<float>> extract_mel_features(const std::vector<float>& audio, int sample_rate) {
    const int n_fft = 400;  // Whisper uses 400 for 16kHz
    const int hop_length = 160;  // 10ms hop for 16kHz
    const int n_mels = 80;
    
    // Resample to 16kHz if needed
    std::vector<float> resampled_audio;
    if (sample_rate != 16000) {
        double ratio = 16000.0 / sample_rate;
        int new_length = static_cast<int>(audio.size() * ratio);
        resampled_audio.resize(new_length);
        for (int i = 0; i < new_length; i++) {
            double src_idx = i / ratio;
            int idx = static_cast<int>(src_idx);
            if (idx + 1 < audio.size()) {
                double frac = src_idx - idx;
                resampled_audio[i] = audio[idx] * (1.0f - frac) + audio[idx + 1] * frac;
            } else if (idx < audio.size()) {
                resampled_audio[i] = audio[idx];
            }
        }
        sample_rate = 16000;
    } else {
        resampled_audio = audio;
    }
    
    // Create mel filter bank
    auto mel_filters = create_mel_filterbank(n_mels, n_fft, sample_rate);
    
    // Calculate number of frames
    int n_frames = (resampled_audio.size() - n_fft) / hop_length + 1;
    n_frames = std::min(n_frames, 3000); // Whisper max frames
    
    // Prepare FFTW
    fftw_complex* fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_plan plan = fftw_plan_dft_1d(n_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    std::vector<std::vector<float>> mel_features(n_frames, std::vector<float>(n_mels, 1e-10f));
    
    for (int frame = 0; frame < n_frames; frame++) {
        // Extract frame with padding
        std::vector<float> frame_data(n_fft, 0.0f);
        int start = frame * hop_length;
        for (int i = 0; i < n_fft && start + i < resampled_audio.size(); i++) {
            frame_data[i] = resampled_audio[start + i];
        }
        
        // Apply window
        apply_hann_window(frame_data);
        
        // Prepare FFT input
        for (int i = 0; i < n_fft; i++) {
            fft_in[i][0] = frame_data[i];
            fft_in[i][1] = 0.0;
        }
        
        // Execute FFT
        fftw_execute(plan);
        
        // Calculate power spectrum
        std::vector<float> power_spectrum(n_fft / 2 + 1);
        for (int i = 0; i < n_fft / 2 + 1; i++) {
            power_spectrum[i] = fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1];
            power_spectrum[i] = std::max(power_spectrum[i], 1e-10f); // Avoid zero values
        }
        
        // Apply mel filters
        for (int m = 0; m < n_mels; m++) {
            float mel_energy = 1e-10f;
            for (int k = 0; k < power_spectrum.size(); k++) {
                mel_energy += mel_filters[m][k] * power_spectrum[k];
            }
            mel_features[frame][m] = mel_energy;
        }
    }
    
    // Cleanup FFTW
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
    
    return mel_features;
}

// Normalize features exactly like Whisper
void normalize_features(std::vector<std::vector<float>>& features) {
    for (auto& frame : features) {
        // Step 1: log10 and find max
        float max_val = -1e20f;
        for (float& f : frame) {
            f = std::max(f, 1e-10f);
            f = std::log10f(f);
            max_val = std::max(f, max_val);
        }
        
        // Step 2: clamp to max - 8
        float min_val = max_val - 8.0f;
        for (float& f : frame) {
            f = std::max(f, min_val);
        }
        
        // Step 3: normalize: (f + 4) / 4
        for (float& f : frame) {
            f = (f + 4.0f) / 4.0f;
        }
    }
}

// Pad or truncate features to target length
std::vector<std::vector<float>> pad_or_trim_features(const std::vector<std::vector<float>>& features, int target_length) {
    std::vector<std::vector<float>> result(target_length, std::vector<float>(80, 0.0f));
    
    int copy_length = std::min(static_cast<int>(features.size()), target_length);
    for (int i = 0; i < copy_length; i++) {
        result[i] = features[i];
    }
    
    return result;
}

int main() {
    try {
        std::cout << "Loading audio file..." << std::endl;
        
        // Load audio
        int sample_rate;
        auto audio = load_audio("testaudio.wav", sample_rate);
        std::cout << "Loaded audio: " << audio.size() << " samples at " << sample_rate << " Hz" << std::endl;
        
        // Extract mel features
        std::cout << "Extracting mel features..." << std::endl;
        auto mel_features = extract_mel_features(audio, sample_rate);
        std::cout << "Extracted " << mel_features.size() << " frames" << std::endl;
        
        // Normalize features
        normalize_features(mel_features);
        
        // Pad to exactly 3000 frames (30 seconds at 16kHz)
        auto padded_features = pad_or_trim_features(mel_features, 3000);
        
        // Load tokens
        std::cout << "Loading tokens..." << std::endl;
        auto tokens = load_tokens("models/whisper-small/small-tokens.txt");
        std::cout << "Loaded " << tokens.size() << " tokens" << std::endl;
        
        // Initialize ONNX Runtime
        std::cout << "Initializing ONNX Runtime..." << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MinimalWhisper");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load encoder
        std::cout << "Loading encoder model..." << std::endl;
        Ort::Session encoder_session(env, L"models/whisper-small/small-encoder.int8.onnx", session_options);
        
        // Load decoder
        std::cout << "Loading decoder model..." << std::endl;
        Ort::Session decoder_session(env, L"models/whisper-small/small-decoder.int8.onnx", session_options);
        
        // Prepare encoder input - transpose to (batch, mel_bins, time)
        std::cout << "Running encoder..." << std::endl;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<float> encoder_input_data(1 * 80 * 3000);
        for (int t = 0; t < 3000; t++) {
            for (int m = 0; m < 80; m++) {
                encoder_input_data[m * 3000 + t] = padded_features[t][m];
            }
        }
        
        std::vector<int64_t> encoder_input_shape = {1, 80, 3000};
        Ort::Value encoder_input = Ort::Value::CreateTensor<float>(
            memory_info, encoder_input_data.data(), encoder_input_data.size(),
            encoder_input_shape.data(), encoder_input_shape.size());
        
        // Run encoder
        std::vector<const char*> encoder_input_names = {"mel"};
        std::vector<const char*> encoder_output_names = {"n_layer_cross_k", "n_layer_cross_v"};
        
        auto encoder_outputs = encoder_session.Run(Ort::RunOptions{nullptr},
            encoder_input_names.data(), &encoder_input, 1,
            encoder_output_names.data(), encoder_output_names.size());
        
        std::cout << "Encoder completed, running decoder..." << std::endl;
        
        // Copy encoder outputs for reuse
        auto encoder_k_shape = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto encoder_v_shape = encoder_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        
        const float* encoder_k_data = encoder_outputs[0].GetTensorData<float>();
        const float* encoder_v_data = encoder_outputs[1].GetTensorData<float>();
        
        int64_t encoder_k_size = 1;
        for (auto dim : encoder_k_shape) encoder_k_size *= dim;
        int64_t encoder_v_size = 1; 
        for (auto dim : encoder_v_shape) encoder_v_size *= dim;
        
        std::vector<float> encoder_k_copy(encoder_k_data, encoder_k_data + encoder_k_size);
        std::vector<float> encoder_v_copy(encoder_v_data, encoder_v_data + encoder_v_size);
        
        // Determine vocab size by running a test
        std::vector<int64_t> test_tokens = {50257}; // Test token
        std::vector<int64_t> test_token_shape = {1, 1};
        
        std::vector<int64_t> kv_cache_shape = {12, 1, 448, 768};
        int64_t kv_cache_size = 12 * 1 * 448 * 768;
        
        std::vector<float> test_self_k(kv_cache_size, 0.0f);
        std::vector<float> test_self_v(kv_cache_size, 0.0f);
        std::vector<int64_t> test_offset = {0};
        std::vector<int64_t> offset_shape = {1};
        
        std::vector<const char*> decoder_input_names = {
            "tokens", "in_n_layer_self_k_cache", "in_n_layer_self_v_cache", 
            "n_layer_cross_k", "n_layer_cross_v", "offset"
        };
        std::vector<const char*> decoder_output_names = {
            "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"
        };
        
        std::vector<Ort::Value> test_inputs;
        test_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, test_tokens.data(), 1, test_token_shape.data(), test_token_shape.size()));
        test_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, test_self_k.data(), kv_cache_size, kv_cache_shape.data(), kv_cache_shape.size()));
        test_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, test_self_v.data(), kv_cache_size, kv_cache_shape.data(), kv_cache_shape.size()));
        test_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, encoder_k_copy.data(), encoder_k_size, encoder_k_shape.data(), encoder_k_shape.size()));
        test_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, encoder_v_copy.data(), encoder_v_size, encoder_v_shape.data(), encoder_v_shape.size()));
        test_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, test_offset.data(), 1, offset_shape.data(), offset_shape.size()));
        
        auto test_outputs = decoder_session.Run(Ort::RunOptions{nullptr},
            decoder_input_names.data(), test_inputs.data(), test_inputs.size(),
            decoder_output_names.data(), decoder_output_names.size());
        
        auto test_logits_shape = test_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = test_logits_shape[2];
        
        std::cout << "Model vocab size: " << vocab_size << std::endl;
        
        // Use proper whisper special tokens
        const int32_t SOT_TOKEN = vocab_size - 6;    // Start of transcript
        const int32_t EOT_TOKEN = vocab_size - 7;    // End of transcript
        const int32_t NO_TIMESTAMPS_TOKEN = vocab_size - 1; // No timestamps
        
        std::cout << "Special tokens: SOT=" << SOT_TOKEN << ", EOT=" << EOT_TOKEN << ", NO_TIMESTAMPS=" << NO_TIMESTAMPS_TOKEN << std::endl;
        
        // Start decoding with proper initial sequence
        std::vector<int64_t> initial_tokens = {SOT_TOKEN, NO_TIMESTAMPS_TOKEN};
        std::vector<int32_t> predicted_tokens;
        
        // Initialize KV caches
        std::vector<float> self_k_cache_data(kv_cache_size, 0.0f);
        std::vector<float> self_v_cache_data(kv_cache_size, 0.0f);
        std::vector<int64_t> offset_data = {0};
        
        // Initial decoder run
        std::vector<int64_t> tokens_shape = {1, static_cast<int64_t>(initial_tokens.size())};
        
        std::vector<Ort::Value> decoder_inputs;
        decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, initial_tokens.data(), initial_tokens.size(),
            tokens_shape.data(), tokens_shape.size()));
        decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, self_k_cache_data.data(), kv_cache_size,
            kv_cache_shape.data(), kv_cache_shape.size()));
        decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, self_v_cache_data.data(), kv_cache_size,
            kv_cache_shape.data(), kv_cache_shape.size()));
        decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, encoder_k_copy.data(), encoder_k_size,
            encoder_k_shape.data(), encoder_k_shape.size()));
        decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, encoder_v_copy.data(), encoder_v_size,
            encoder_v_shape.data(), encoder_v_shape.size()));
        decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, offset_data.data(), 1, offset_shape.data(), offset_shape.size()));
        
        auto decoder_outputs = decoder_session.Run(Ort::RunOptions{nullptr},
            decoder_input_names.data(), decoder_inputs.data(), decoder_inputs.size(),
            decoder_output_names.data(), decoder_output_names.size());
        
        // Get first prediction
        const float* logits_data = decoder_outputs[0].GetTensorData<float>();
        auto logits_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        const float* last_logits = logits_data + (logits_shape[1] - 1) * vocab_size;
        int32_t next_token = static_cast<int32_t>(
            std::distance(last_logits, std::max_element(last_logits, last_logits + vocab_size)));
        
        // Copy updated KV caches
        const float* updated_self_k = decoder_outputs[1].GetTensorData<float>();
        const float* updated_self_v = decoder_outputs[2].GetTensorData<float>();
        std::copy(updated_self_k, updated_self_k + kv_cache_size, self_k_cache_data.begin());
        std::copy(updated_self_v, updated_self_v + kv_cache_size, self_v_cache_data.begin());
        
        offset_data[0] = initial_tokens.size();
        
        // Iterative decoding
        int max_tokens = 224; // Reasonable limit for whisper small
        for (int step = 0; step < max_tokens; step++) {
            if (next_token == EOT_TOKEN) {
                std::cout << "EOT token detected, ending generation" << std::endl;
                break;
            }
            
            // Skip special tokens in output but still process them
            if (tokens.find(next_token) != tokens.end() || next_token < 50000) {
                predicted_tokens.push_back(next_token);
            }
            
            // Prepare single token input
            std::vector<int64_t> single_token = {next_token};
            std::vector<int64_t> single_token_shape = {1, 1};
            
            std::vector<Ort::Value> next_decoder_inputs;
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, single_token.data(), 1,
                single_token_shape.data(), single_token_shape.size()));
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, self_k_cache_data.data(), kv_cache_size,
                kv_cache_shape.data(), kv_cache_shape.size()));
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, self_v_cache_data.data(), kv_cache_size,
                kv_cache_shape.data(), kv_cache_shape.size()));
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, encoder_k_copy.data(), encoder_k_size,
                encoder_k_shape.data(), encoder_k_shape.size()));
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, encoder_v_copy.data(), encoder_v_size,
                encoder_v_shape.data(), encoder_v_shape.size()));
            next_decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, offset_data.data(), 1, offset_shape.data(), offset_shape.size()));
            
            decoder_outputs = decoder_session.Run(Ort::RunOptions{nullptr},
                decoder_input_names.data(), next_decoder_inputs.data(), next_decoder_inputs.size(),
                decoder_output_names.data(), decoder_output_names.size());
            
            // Get next prediction
            logits_data = decoder_outputs[0].GetTensorData<float>();
            next_token = static_cast<int32_t>(
                std::distance(logits_data, std::max_element(logits_data, logits_data + vocab_size)));
            
            // Update KV caches
            updated_self_k = decoder_outputs[1].GetTensorData<float>();
            updated_self_v = decoder_outputs[2].GetTensorData<float>();
            std::copy(updated_self_k, updated_self_k + kv_cache_size, self_k_cache_data.begin());
            std::copy(updated_self_v, updated_self_v + kv_cache_size, self_v_cache_data.begin());
            
            offset_data[0]++;
        }
        
        // Convert tokens to text
        std::cout << "Converting " << predicted_tokens.size() << " tokens to text..." << std::endl;
        std::string result_text;
        
        for (int32_t token_id : predicted_tokens) {
            if (tokens.find(token_id) != tokens.end()) {
                result_text += tokens[token_id];
            }
        }
        
        std::cout << "Result: " << result_text << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}