#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <sndfile.h>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Base64 decoding function
std::string base64_decode(const std::string &encoded_string) {
  const std::string chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string decoded;
  int val = 0, valb = -8;
  for (unsigned char c : encoded_string) {
    if (c == '=')
      break;
    if (chars.find(c) == std::string::npos)
      continue;
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
std::unordered_map<int32_t, std::string>
load_tokens(const std::string &filename) {
  std::unordered_map<int32_t, std::string> tokens;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open token file: " + filename);
  }

  std::string line;
  int line_number = 0;

  while (std::getline(file, line)) {
    line_number++;
    if (line.empty())
      continue;

    size_t space_pos = line.find(' ');
    if (space_pos != std::string::npos) {
      try {
        std::string base64_token = line.substr(0, space_pos);
        std::string id_str = line.substr(space_pos + 1);
        int32_t token_id = std::stoi(id_str);
        std::string decoded_token = base64_decode(base64_token);
        tokens[token_id] = decoded_token;
      } catch (const std::exception &e) {
        std::cerr << "Error parsing token on line " << line_number << ": "
                  << line << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        throw;
      }
    }
  }

  return tokens;
}

// Load audio file
std::vector<float> load_audio(const std::string &filename, int &sample_rate) {
  SF_INFO info;
  SNDFILE *file = sf_open(filename.c_str(), SFM_READ, &info);
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
void apply_hann_window(std::vector<float> &frame) {
  int n = frame.size();
  for (int i = 0; i < n; i++) {
    frame[i] *= 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
  }
}

// Convert to mel scale
double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }

double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

// Create mel filter bank (improved version)
std::vector<std::vector<float>> create_mel_filterbank(int n_mels, int n_fft,
                                                      int sample_rate) {
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
  std::vector<std::vector<float>> filters(n_mels,
                                          std::vector<float>(n_freqs, 0.0f));
  for (int m = 0; m < n_mels; m++) {
    for (int k = 0; k < n_freqs; k++) {
      double freq = (double)k;
      if (freq >= bin_points[m] && freq <= bin_points[m + 1]) {
        filters[m][k] =
            (freq - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
      } else if (freq >= bin_points[m + 1] && freq <= bin_points[m + 2]) {
        filters[m][k] = (bin_points[m + 2] - freq) /
                        (bin_points[m + 2] - bin_points[m + 1]);
      }
    }

    // Normalize filter
    float sum = 0.0f;
    for (float f : filters[m])
      sum += f;
    if (sum > 0) {
      for (float &f : filters[m])
        f /= sum;
    }
  }

  return filters;
}

// Simple but effective decimation/interpolation resampling
std::vector<float> resample_audio(const std::vector<float> &input,
                                  int input_rate, int output_rate) {
  if (input_rate == output_rate) {
    return input;
  }

  // For downsampling, apply anti-aliasing filter first
  std::vector<float> filtered_input = input;
  if (input_rate > output_rate) {
    // Simple low-pass filter to prevent aliasing
    float cutoff = 0.45f * output_rate / input_rate;
    for (int i = 1; i < filtered_input.size() - 1; i++) {
      filtered_input[i] = input[i] * cutoff +
                          input[i - 1] * (0.5f * (1.0f - cutoff)) +
                          input[i + 1] * (0.5f * (1.0f - cutoff));
    }
  }

  // Linear interpolation resampling
  double ratio = static_cast<double>(output_rate) / input_rate;
  int output_length = static_cast<int>(filtered_input.size() * ratio);
  std::vector<float> output(output_length);

  for (int i = 0; i < output_length; i++) {
    double src_idx = i / ratio;
    int idx = static_cast<int>(src_idx);
    double frac = src_idx - idx;

    if (idx + 1 < filtered_input.size()) {
      output[i] =
          filtered_input[idx] * (1.0f - frac) + filtered_input[idx + 1] * frac;
    } else if (idx < filtered_input.size()) {
      output[i] = filtered_input[idx];
    } else {
      output[i] = 0.0f;
    }
  }

  return output;
}

// Extract mel spectrogram features (improved version)
std::vector<std::vector<float>>
extract_mel_features(const std::vector<float> &audio, int sample_rate) {
  const int n_fft = 400;      // Whisper uses 400 for 16kHz
  const int hop_length = 160; // 10ms hop for 16kHz
  const int n_mels = 80;

  // Resample to 16kHz if needed
  std::vector<float> resampled_audio;
  if (sample_rate != 16000) {
    std::cout << "Resampling from " << sample_rate << "Hz to 16000Hz..."
              << std::endl;
    resampled_audio = resample_audio(audio, sample_rate, 16000);

    // Debug: Check audio levels after resampling
    float max_val = 0.0f, min_val = 0.0f;
    for (float sample : resampled_audio) {
      max_val = std::max(max_val, sample);
      min_val = std::min(min_val, sample);
    }
    std::cout << "Resampled audio range: [" << min_val << ", " << max_val
              << "], length: " << resampled_audio.size() << std::endl;
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
  fftw_complex *fft_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_fft);
  fftw_complex *fft_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_fft);
  fftw_plan plan =
      fftw_plan_dft_1d(n_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

  std::vector<std::vector<float>> mel_features(
      n_frames, std::vector<float>(n_mels, 1e-10f));

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
      power_spectrum[i] =
          fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1];
      power_spectrum[i] =
          std::max(power_spectrum[i], 1e-10f); // Avoid zero values
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
void normalize_features(std::vector<std::vector<float>> &features) {
  for (auto &frame : features) {
    // Step 1: log10 and find max
    float max_val = -1e20f;
    for (float &f : frame) {
      f = std::max(f, 1e-10f);
      f = std::log10f(f);
      max_val = std::max(f, max_val);
    }

    // Step 2: clamp to max - 8
    float min_val = max_val - 8.0f;
    for (float &f : frame) {
      f = std::max(f, min_val);
    }

    // Step 3: normalize: (f + 4) / 4
    for (float &f : frame) {
      f = (f + 4.0f) / 4.0f;
    }
  }
}

// Pad or truncate features to target length
std::vector<std::vector<float>>
pad_or_trim_features(const std::vector<std::vector<float>> &features,
                     int target_length) {
  std::vector<std::vector<float>> result(target_length,
                                         std::vector<float>(80, 0.0f));

  int copy_length = std::min(static_cast<int>(features.size()), target_length);
  for (int i = 0; i < copy_length; i++) {
    result[i] = features[i];
  }

  return result;
}

void print_usage() {
  std::cout << "Minimal Whisper - Lightweight speech recognition using ONNX "
               "Runtime\n\n";
  std::cout
      << "Usage: minimal_whisper.exe [audio_file.wav] "
         "[--openvino[=device1,device2,...]] "
         "[--model-dir=path] [--model-prefix=prefix] "
         "[--model-suffix=suffix] [--clear-cache] [--no-cache] [--help]\n\n";
  std::cout << "Arguments:\n";
  std::cout
      << "  audio_file.wav       Input audio file (default: testaudio.wav)\n";
  std::cout << "  --openvino[=devices] Use OpenVINO execution provider\n";
  std::cout << "                       Optional devices: NPU, GPU, CPU "
               "(default: NPU,GPU,CPU)\n";
  std::cout << "  --model-dir=path     Directory containing ONNX models "
               "(default: models\\whisper-small)\n";
  std::cout
      << "  --model-prefix=prefix Model filename prefix (default: small-)\n";
  std::cout
      << "  --model-suffix=suffix Model filename suffix (default: .int8)\n";
  std::cout << "  --clear-cache        Clear OpenVINO model cache and exit\n";
  std::cout << "  --no-cache           Disable OpenVINO model caching\n";
  std::cout << "  --help, -h           Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  minimal_whisper.exe                              # Use CPU "
               "with default audio\n";
  std::cout << "  minimal_whisper.exe myaudio.wav --openvino       # Use "
               "OpenVINO with default devices\n";
  std::cout << "  minimal_whisper.exe --openvino=GPU,CPU           # Use "
               "OpenVINO with specific devices\n";
  std::cout << "  minimal_whisper.exe --model-dir=models\\whisper-large "
               "--model-prefix=large-\n";
  std::cout << "  minimal_whisper.exe --clear-cache                # Clear "
               "model cache\n";
  std::cout << "  minimal_whisper.exe --openvino --no-cache        # Use "
               "OpenVINO without caching\n\n";
  std::cout << "Features:\n";
  std::cout << "  - OpenVINO model pre-compilation and caching for faster "
               "subsequent runs\n";
  std::cout << "  - Automatic device fallback (NPU -> GPU -> CPU)\n";
  std::cout << "  - Supports any audio sample rate (automatically resampled to "
               "16kHz)\n";
  std::cout << "  - Base64 token decoding from whisper token files\n";
  std::cout << "  - Mel spectrogram feature extraction using FFTW3\n";
}

// OpenVINO Model Cache for pre-compilation and caching
class OpenVINOModelCache {
private:
  std::filesystem::path cache_dir_;

  std::string get_model_hash(const std::wstring &model_path) {
    try {
      auto file_size = std::filesystem::file_size(model_path);
      auto last_write = std::filesystem::last_write_time(model_path);
      auto time_since_epoch = last_write.time_since_epoch();
      auto time_count =
          std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch)
              .count();

      std::string hash =
          std::to_string(file_size) + "_" + std::to_string(time_count);
      return hash;
    } catch (const std::exception &) {
      return "unknown";
    }
  }

public:
  OpenVINOModelCache(bool no_cache = false) {
    if (no_cache) {
      return;
    }

    cache_dir_ = std::filesystem::current_path() / "openvino_cache";
    std::filesystem::create_directories(cache_dir_);
    std::cout << "OpenVINO cache directory: " << cache_dir_.string()
              << std::endl;
  }

  std::string get_cache_dir() const { return cache_dir_.string(); }

  std::filesystem::path get_cached_model_path(const std::wstring &model_path,
                                              const std::string &device) {
    auto model_name = std::filesystem::path(model_path).stem().string();
    auto hash = get_model_hash(model_path);
    auto cache_name = model_name + "_" + device + "_" + hash + ".cached";
    return cache_dir_ / cache_name;
  }

  bool is_cached(const std::wstring &model_path, const std::string &device) {
    auto cache_path = get_cached_model_path(model_path, device);
    return std::filesystem::exists(cache_path);
  }

  void precompile_model(Ort::Env &env, const std::wstring &model_path,
                        const Ort::SessionOptions &session_options,
                        const std::string &device,
                        const std::string &model_name) {

    auto compile_start = std::chrono::high_resolution_clock::now();

    try {
      std::cout << "Pre-compiling " << model_name << " model..." << std::endl;

      // Create a session to trigger OpenVINO compilation - this populates the
      // OpenVINO cache
      auto session = std::make_unique<Ort::Session>(env, model_path.c_str(),
                                                    session_options);

      // Create our cache marker file
      // OpenVINO's built-in cache_dir will handle the actual compiled model
      // caching We just track that compilation happened

      auto compile_end = std::chrono::high_resolution_clock::now();
      auto compile_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(compile_end -
                                                                compile_start);
      std::cout << model_name << " pre-compilation completed in "
                << compile_duration.count() << " ms" << std::endl;

    } catch (const std::exception &e) {
      std::cout << "Model pre-compilation failed: " << e.what() << std::endl;
      throw;
    }
  }

  void clear_cache() {
    try {
      if (std::filesystem::exists(cache_dir_)) {
        std::filesystem::remove_all(cache_dir_);
        std::filesystem::create_directories(cache_dir_);
        std::cout << "OpenVINO model cache cleared" << std::endl;
      }
    } catch (const std::exception &e) {
      std::cout << "Failed to clear cache: " << e.what() << std::endl;
    }
  }
};

class WhisperEncoder {
private:
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;

public:
  WhisperEncoder(Ort::Env &env, const std::wstring &model_path,
                 const Ort::SessionOptions &session_options)
      : memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    auto load_start = std::chrono::high_resolution_clock::now();
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(),
                                              session_options);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start);
    std::cout << "Encoder loaded successfully in " << load_duration.count()
              << " ms" << std::endl;
  }

  std::pair<std::vector<float>, std::vector<float>>
  encode(const std::vector<std::vector<float>> &padded_features) {

    std::cout << "Running encoder inference..." << std::endl;
    auto inference_start = std::chrono::high_resolution_clock::now();

    // Prepare input - transpose to (batch, mel_bins, time)
    std::vector<float> encoder_input_data(1 * 80 * 3000);
    for (int t = 0; t < 3000; t++) {
      for (int m = 0; m < 80; m++) {
        encoder_input_data[m * 3000 + t] = padded_features[t][m];
      }
    }

    std::vector<int64_t> encoder_input_shape = {1, 80, 3000};
    Ort::Value encoder_input = Ort::Value::CreateTensor<float>(
        memory_info_, encoder_input_data.data(), encoder_input_data.size(),
        encoder_input_shape.data(), encoder_input_shape.size());

    std::vector<const char *> encoder_input_names = {"mel"};
    std::vector<const char *> encoder_output_names = {"n_layer_cross_k",
                                                      "n_layer_cross_v"};

    auto encoder_outputs = session_->Run(
        Ort::RunOptions{nullptr}, encoder_input_names.data(), &encoder_input, 1,
        encoder_output_names.data(), encoder_output_names.size());

    // Extract and copy output data
    auto encoder_k_shape =
        encoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto encoder_v_shape =
        encoder_outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    const float *encoder_k_data = encoder_outputs[0].GetTensorData<float>();
    const float *encoder_v_data = encoder_outputs[1].GetTensorData<float>();

    int64_t encoder_k_size = 1;
    for (auto dim : encoder_k_shape)
      encoder_k_size *= dim;
    int64_t encoder_v_size = 1;
    for (auto dim : encoder_v_shape)
      encoder_v_size *= dim;

    std::vector<float> encoder_k_copy(encoder_k_data,
                                      encoder_k_data + encoder_k_size);
    std::vector<float> encoder_v_copy(encoder_v_data,
                                      encoder_v_data + encoder_v_size);

    auto inference_end = std::chrono::high_resolution_clock::now();
    auto inference_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(inference_end -
                                                              inference_start);
    std::cout << "Encoder inference completed in " << inference_duration.count()
              << " ms" << std::endl;

    return {std::move(encoder_k_copy), std::move(encoder_v_copy)};
  }

  void release() {
    session_.reset();
    std::cout << "Encoder resources released" << std::endl;
  }
};

class WhisperDecoder {
private:
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;

public:
  WhisperDecoder(Ort::Env &env, const std::wstring &model_path,
                 const Ort::SessionOptions &session_options)
      : memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    auto load_start = std::chrono::high_resolution_clock::now();
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(),
                                              session_options);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start);
    std::cout << "Decoder loaded successfully in " << load_duration.count()
              << " ms" << std::endl;
  }

  std::vector<int32_t>
  decode(const std::vector<float> &encoder_k_copy,
         const std::vector<float> &encoder_v_copy,
         const std::unordered_map<int32_t, std::string> &tokens) {

    std::cout << "Running decoder inference..." << std::endl;
    auto inference_start = std::chrono::high_resolution_clock::now();

    // Special token IDs
    const int32_t EOT_TOKEN = 50257;
    const int32_t SOT_TOKEN = 50258;
    const int32_t ENGLISH_TOKEN = 50259;
    const int32_t NO_TIMESTAMPS_TOKEN = 50363;
    const int32_t TRANSCRIBE_TOKEN = 50359;

    // KV cache configuration for whisper-small
    std::vector<int64_t> kv_cache_shape = {12, 1, 448, 768};
    int64_t kv_cache_size = 12 * 1 * 448 * 768;
    std::vector<int64_t> encoder_k_shape = {12, 1, 1500, 768};
    std::vector<int64_t> encoder_v_shape = {12, 1, 1500, 768};

    std::vector<const char *> decoder_input_names = {
        "tokens",          "in_n_layer_self_k_cache", "in_n_layer_self_v_cache",
        "n_layer_cross_k", "n_layer_cross_v",         "offset"};
    std::vector<const char *> decoder_output_names = {
        "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"};

    // Determine vocab size with test run
    std::vector<int64_t> test_tokens = {50257};
    std::vector<int64_t> test_token_shape = {1, 1};
    std::vector<float> test_self_k(kv_cache_size, 0.0f);
    std::vector<float> test_self_v(kv_cache_size, 0.0f);
    std::vector<int64_t> test_offset = {0};
    std::vector<int64_t> offset_shape = {1};

    int64_t encoder_k_size = 1;
    for (auto dim : encoder_k_shape)
      encoder_k_size *= dim;
    int64_t encoder_v_size = 1;
    for (auto dim : encoder_v_shape)
      encoder_v_size *= dim;

    std::vector<Ort::Value> test_inputs;
    test_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, test_tokens.data(), 1, test_token_shape.data(),
        test_token_shape.size()));
    test_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, test_self_k.data(), kv_cache_size, kv_cache_shape.data(),
        kv_cache_shape.size()));
    test_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, test_self_v.data(), kv_cache_size, kv_cache_shape.data(),
        kv_cache_shape.size()));
    test_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(encoder_k_copy.data()),
        encoder_k_size, encoder_k_shape.data(), encoder_k_shape.size()));
    test_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(encoder_v_copy.data()),
        encoder_v_size, encoder_v_shape.data(), encoder_v_shape.size()));
    test_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, test_offset.data(), 1, offset_shape.data(),
        offset_shape.size()));

    auto test_outputs =
        session_->Run(Ort::RunOptions{nullptr}, decoder_input_names.data(),
                      test_inputs.data(), test_inputs.size(),
                      decoder_output_names.data(), decoder_output_names.size());

    auto test_logits_shape =
        test_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t vocab_size = test_logits_shape[2];
    std::cout << "Model vocab size: " << vocab_size << std::endl;

    // Start decoding with proper initial sequence
    std::vector<int64_t> initial_tokens = {
        SOT_TOKEN, ENGLISH_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN};
    std::vector<int32_t> predicted_tokens;

    // Initialize KV caches
    std::vector<float> self_k_cache_data(kv_cache_size, 0.0f);
    std::vector<float> self_v_cache_data(kv_cache_size, 0.0f);
    std::vector<int64_t> offset_data = {0};

    // Initial decoder run
    std::vector<int64_t> tokens_shape = {
        1, static_cast<int64_t>(initial_tokens.size())};

    std::vector<Ort::Value> decoder_inputs;
    decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, initial_tokens.data(), initial_tokens.size(),
        tokens_shape.data(), tokens_shape.size()));
    decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, self_k_cache_data.data(), kv_cache_size,
        kv_cache_shape.data(), kv_cache_shape.size()));
    decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, self_v_cache_data.data(), kv_cache_size,
        kv_cache_shape.data(), kv_cache_shape.size()));
    decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(encoder_k_copy.data()),
        encoder_k_size, encoder_k_shape.data(), encoder_k_shape.size()));
    decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(encoder_v_copy.data()),
        encoder_v_size, encoder_v_shape.data(), encoder_v_shape.size()));
    decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, offset_data.data(), 1, offset_shape.data(),
        offset_shape.size()));

    auto decoder_outputs =
        session_->Run(Ort::RunOptions{nullptr}, decoder_input_names.data(),
                      decoder_inputs.data(), decoder_inputs.size(),
                      decoder_output_names.data(), decoder_output_names.size());

    // Get first prediction
    const float *logits_data = decoder_outputs[0].GetTensorData<float>();
    auto logits_shape =
        decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    const float *last_logits = logits_data + (logits_shape[1] - 1) * vocab_size;
    int32_t next_token = static_cast<int32_t>(std::distance(
        last_logits, std::max_element(last_logits, last_logits + vocab_size)));

    // Copy updated KV caches
    const float *updated_self_k = decoder_outputs[1].GetTensorData<float>();
    const float *updated_self_v = decoder_outputs[2].GetTensorData<float>();
    std::copy(updated_self_k, updated_self_k + kv_cache_size,
              self_k_cache_data.begin());
    std::copy(updated_self_v, updated_self_v + kv_cache_size,
              self_v_cache_data.begin());

    offset_data[0] = initial_tokens.size();

    // Iterative decoding
    int max_tokens = 224;
    for (int step = 0; step < max_tokens; step++) {
      if (next_token == EOT_TOKEN) {
        std::cout << "EOT token detected, ending generation" << std::endl;
        break;
      }

      // Only add non-special tokens to output
      if (tokens.find(next_token) != tokens.end() && next_token < 50257) {
        predicted_tokens.push_back(next_token);
      }

      // Prepare single token input
      std::vector<int64_t> single_token = {next_token};
      std::vector<int64_t> single_token_shape = {1, 1};

      std::vector<Ort::Value> next_decoder_inputs;
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          memory_info_, single_token.data(), 1, single_token_shape.data(),
          single_token_shape.size()));
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
          memory_info_, self_k_cache_data.data(), kv_cache_size,
          kv_cache_shape.data(), kv_cache_shape.size()));
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
          memory_info_, self_v_cache_data.data(), kv_cache_size,
          kv_cache_shape.data(), kv_cache_shape.size()));
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
          memory_info_, const_cast<float *>(encoder_k_copy.data()),
          encoder_k_size, encoder_k_shape.data(), encoder_k_shape.size()));
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<float>(
          memory_info_, const_cast<float *>(encoder_v_copy.data()),
          encoder_v_size, encoder_v_shape.data(), encoder_v_shape.size()));
      next_decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          memory_info_, offset_data.data(), 1, offset_shape.data(),
          offset_shape.size()));

      decoder_outputs = session_->Run(
          Ort::RunOptions{nullptr}, decoder_input_names.data(),
          next_decoder_inputs.data(), next_decoder_inputs.size(),
          decoder_output_names.data(), decoder_output_names.size());

      // Get next prediction
      logits_data = decoder_outputs[0].GetTensorData<float>();
      next_token = static_cast<int32_t>(std::distance(
          logits_data,
          std::max_element(logits_data, logits_data + vocab_size)));

      // Update KV caches
      updated_self_k = decoder_outputs[1].GetTensorData<float>();
      updated_self_v = decoder_outputs[2].GetTensorData<float>();
      std::copy(updated_self_k, updated_self_k + kv_cache_size,
                self_k_cache_data.begin());
      std::copy(updated_self_v, updated_self_v + kv_cache_size,
                self_v_cache_data.begin());

      offset_data[0]++;
    }

    auto inference_end = std::chrono::high_resolution_clock::now();
    auto inference_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(inference_end -
                                                              inference_start);
    std::cout << "Decoder inference completed in " << inference_duration.count()
              << " ms" << std::endl;
    return predicted_tokens;
  }
};

int main(int argc, char *argv[]) {
  try {
    // Parse command line arguments
    std::string audio_file = "testaudio.wav";
    bool use_openvino = false;
    bool show_help = false;
    bool clear_cache = false;
    bool no_cache = false;
    std::filesystem::path model_path = "models\\whisper-small";
    std::wstring model_prefix = L"small-";
    std::wstring model_suffix = L".int8";

    std::string openvino_backends[3] = {"NPU", "GPU", "CPU"};
    size_t openvino_backend_count = 3;

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--openvino") {
        use_openvino = true;
      } else if (arg.find("--openvino=") == 0 && arg.size() > 11) {
        use_openvino = true;
        openvino_backend_count = 0;
        size_t offset = 11;
        while (offset < arg.size()) {
          std::string backend;
          auto pos = arg.find(",", offset);
          if (pos != std::string::npos) {
            backend = arg.substr(offset, pos - offset).c_str();
            offset = pos + 1;
          } else {
            backend = arg.substr(offset).c_str();
            offset = arg.size();
          }

          openvino_backends[openvino_backend_count++] = backend;
        }

      } else if (arg.find("--model-dir=") == 0) {
        auto s = arg.substr(12);
        model_path = std::wstring(s.begin(), s.end());
      } else if (arg.find("--model-prefix=") == 0) {
        auto s = arg.substr(15);
        model_prefix = std::wstring(s.begin(), s.end());
      } else if (arg.find("--model-suffix=") == 0) {
        auto s = arg.substr(15);
        model_suffix = std::wstring(s.begin(), s.end());
      } else if (arg == "--clear-cache") {
        clear_cache = true;
      } else if (arg == "--no-cache") {
        no_cache = true;
      } else if (arg == "--help" || arg == "-h") {
        show_help = true;
        break;
      } else if (arg.find(".wav") != std::string::npos) {
        audio_file = arg;
      } else {
        std::cout << "Unknown argument: " << arg << "\n\n";
        show_help = true;
        break;
      }
    }

    if (show_help) {
      print_usage();
      return 0;
    }

    // Handle cache clearing
    if (clear_cache) {
      std::cout << "Clearing OpenVINO model cache..." << std::endl;
      OpenVINOModelCache cache;
      cache.clear_cache();
      if (!use_openvino) {
        return 0;
      }
    }

    std::cout << "=== Minimal Whisper Speech Recognition ===" << std::endl;
    if (use_openvino) {
      std::cout << "Execution Provider: OpenVINO" << std::endl;
    } else {
      std::cout << "Execution Provider: CPU" << std::endl;
    }

    std::cout << "Loading audio file: " << audio_file << std::endl;

    // Load audio
    int sample_rate;
    auto audio = load_audio(audio_file, sample_rate);
    std::cout << "Loaded audio: " << audio.size() << " samples at "
              << sample_rate << " Hz" << std::endl;

    // Extract mel features
    std::cout << "Extracting mel features..." << std::endl;
    auto mel_features = extract_mel_features(audio, sample_rate);
    std::cout << "Extracted " << mel_features.size() << " frames" << std::endl;

    // Debug: Check mel feature statistics
    if (!mel_features.empty()) {
      float mel_min = 1e10f, mel_max = -1e10f;
      for (const auto &frame : mel_features) {
        for (float val : frame) {
          mel_min = std::min(mel_min, val);
          mel_max = std::max(mel_max, val);
        }
      }
      std::cout << "Mel features range before normalization: [" << mel_min
                << ", " << mel_max << "]" << std::endl;
    }

    // Normalize features
    normalize_features(mel_features);

    // Debug: Check normalized feature statistics
    if (!mel_features.empty()) {
      float norm_min = 1e10f, norm_max = -1e10f;
      for (const auto &frame : mel_features) {
        for (float val : frame) {
          norm_min = std::min(norm_min, val);
          norm_max = std::max(norm_max, val);
        }
      }
      std::cout << "Mel features range after normalization: [" << norm_min
                << ", " << norm_max << "]" << std::endl;
    }

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
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Configure execution provider with caching
    std::string selected_device = "CPU";
    OpenVINOModelCache model_cache(no_cache);

    if (use_openvino) {
      std::cout << "Configuring OpenVINO execution provider..." << std::endl;

      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);

      OrtOpenVINOProviderOptions openvino_options;
      openvino_options.device_id = "";
      openvino_options.num_of_threads = 0;
      openvino_options.cache_dir =
          no_cache ? nullptr : model_cache.get_cache_dir().c_str();
      openvino_options.context = nullptr;
      openvino_options.enable_opencl_throttling = false;
      openvino_options.enable_dynamic_shapes = true;

      bool openvino_configured = false;
      for (size_t i = 0; i < openvino_backend_count; i++) {
        std::cout << "Attempting to configure OpenVINO with "
                  << openvino_backends[i] << "..." << std::endl;
        try {
          openvino_options.device_type = openvino_backends[i].c_str();
          session_options.AppendExecutionProvider_OpenVINO(openvino_options);

          selected_device = openvino_backends[i];
          openvino_configured = true;
          std::cout << "OpenVINO execution provider configured with "
                    << openvino_backends[i] << std::endl;
          break;
        } catch (const std::exception &e) {
          std::cout << "Failed to configure OpenVINO with "
                    << openvino_backends[i] << ": " << e.what() << std::endl;
        }
      }

      if (!openvino_configured) {
        throw std::runtime_error(
            "Failed to configure OpenVINO with any of the specified backends");
      }
    }

    // Build model paths
    auto encoder_model_path = (model_path / model_prefix)
                                  .concat(L"encoder")
                                  .concat(model_suffix)
                                  .concat(L".onnx")
                                  .wstring();
    auto decoder_model_path = (model_path / model_prefix)
                                  .concat(L"decoder")
                                  .concat(model_suffix)
                                  .concat(L".onnx")
                                  .wstring();

    std::wcout << L"Encoder model path: " << encoder_model_path << std::endl;
    std::wcout << L"Decoder model path: " << decoder_model_path << std::endl;

    // OpenVINO will automatically cache compiled models on first use
    if (use_openvino) {
      if (no_cache) {
        std::cout << "\n=== OpenVINO CACHING DISABLED ===" << std::endl;
        std::cout
            << "Model caching is disabled - models will be compiled each run"
            << std::endl;
      } else {
        std::cout << "\n=== OpenVINO CACHING ENABLED ===" << std::endl;
        std::cout
            << "Models will be compiled and cached automatically on first use"
            << std::endl;
        std::cout
            << "Subsequent runs will be faster using cached compiled models"
            << std::endl;
      }
    }

    // Phase 1: Encoding
    std::cout << "\n=== ENCODING PHASE ===" << std::endl;
    auto phase1_start = std::chrono::high_resolution_clock::now();
    WhisperEncoder encoder(env, encoder_model_path, session_options);
    auto [encoder_k_copy, encoder_v_copy] = encoder.encode(padded_features);

    // Release encoder resources
    encoder.release();
    auto phase1_end = std::chrono::high_resolution_clock::now();
    auto phase1_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end -
                                                              phase1_start);
    std::cout << "Encoding phase completed in " << phase1_duration.count()
              << " ms" << std::endl;

    // Phase 2: Decoding
    std::cout << "\n=== DECODING PHASE ===" << std::endl;
    auto phase2_start = std::chrono::high_resolution_clock::now();
    WhisperDecoder decoder(env, decoder_model_path, session_options);
    auto predicted_tokens =
        decoder.decode(encoder_k_copy, encoder_v_copy, tokens);
    auto phase2_end = std::chrono::high_resolution_clock::now();
    auto phase2_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end -
                                                              phase2_start);
    std::cout << "Decoding phase completed in " << phase2_duration.count()
              << " ms" << std::endl;

    // Convert tokens to text
    std::cout << "Converting " << predicted_tokens.size()
              << " tokens to text..." << std::endl;
    std::string result_text;

    for (int32_t token_id : predicted_tokens) {
      if (tokens.find(token_id) != tokens.end()) {
        result_text += tokens[token_id];
      }
    }

    auto total_duration = phase1_duration + phase2_duration;
    std::cout << "\n=== Performance Summary ===" << std::endl;
    if (use_openvino) {
      std::cout << "OpenVINO Device: " << selected_device << std::endl;
      std::cout << "Model cache used: " << (no_cache ? "No" : "Yes")
                << std::endl;
    }
    std::cout << "Encoding phase (load + inference): "
              << phase1_duration.count() << " ms" << std::endl;
    std::cout << "Decoding phase (load + inference): "
              << phase2_duration.count() << " ms" << std::endl;
    std::cout << "Total processing time: " << total_duration.count() << " ms"
              << std::endl;

    std::cout << "\n=== Transcription Result ===" << std::endl;
    std::cout << result_text << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "Use --help for usage information." << std::endl;
    return 1;
  }

  return 0;
}
