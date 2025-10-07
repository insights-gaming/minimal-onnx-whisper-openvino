#include <algorithm>
#include <chrono>
#include <cmath>
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

// Load configuration from file
struct ModelConfig {
  int sampling_rate = 16000;
  int num_labels = 3; // Valence, Arousal, Dominance by default
  std::vector<std::string> label_names;
};

ModelConfig load_config(const std::string &filename) {
  ModelConfig config;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Warning: Could not open config file: " << filename << std::endl;
    std::cout << "Using default configuration (16kHz, 3 labels: V/A/D)" << std::endl;
    config.label_names = {"Valence", "Arousal", "Dominance"};
    return config;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;

    size_t equals_pos = line.find('=');
    if (equals_pos != std::string::npos) {
      std::string key = line.substr(0, equals_pos);
      std::string value = line.substr(equals_pos + 1);

      if (key == "sampling_rate") {
        config.sampling_rate = std::stoi(value);
      } else if (key == "num_labels") {
        config.num_labels = std::stoi(value);
      } else if (key.find("label_") == 0) {
        config.label_names.push_back(value);
      }
    }
  }

  // If no labels were loaded, use defaults
  if (config.label_names.empty()) {
    config.label_names = {"Valence", "Arousal", "Dominance"};
  }

  return config;
}

void print_usage() {
  std::cout << "Usage: minimal_voice_analysis [options] <audio_file>" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --model <path>       Path to ONNX model file (default: models/vad/vad_model.onnx)" << std::endl;
  std::cout << "  --config <path>      Path to config file (default: models/vad/config.txt)" << std::endl;
  std::cout << "  --openvino           Use OpenVINO execution provider" << std::endl;
  std::cout << "  --no-cache           Disable OpenVINO model caching" << std::endl;
  std::cout << "  --help               Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  minimal_voice_analysis audio.wav" << std::endl;
  std::cout << "  minimal_voice_analysis --openvino audio.wav" << std::endl;
  std::cout << "  minimal_voice_analysis --model custom.onnx audio.wav" << std::endl;
}

// OpenVINO Model Cache for pre-compilation and caching
class OpenVINOModelCache {
private:
  std::filesystem::path cache_dir_;
  bool enabled_;

public:
  OpenVINOModelCache(bool disable_cache = false)
      : cache_dir_("openvino_cache"), enabled_(!disable_cache) {
    if (enabled_) {
      if (!std::filesystem::exists(cache_dir_)) {
        std::filesystem::create_directory(cache_dir_);
      }
    }
  }

  std::string get_cache_dir() const { return cache_dir_.string(); }

  bool is_enabled() const { return enabled_; }
};

// Voice Analysis Inference class
class VoiceAnalyzer {
private:
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  ModelConfig config_;

public:
  VoiceAnalyzer(Ort::Env &env, const std::wstring &model_path,
                const Ort::SessionOptions &session_options,
                const ModelConfig &config)
      : memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        config_(config) {
    auto load_start = std::chrono::high_resolution_clock::now();
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(),
                                              session_options);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start);
    std::cout << "Model loaded successfully in " << load_duration.count()
              << " ms" << std::endl;
  }

  std::vector<float> analyze(const std::vector<float> &audio_data) {
    // Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(audio_data.size())};
    
    // Copy audio data (non-const)
    std::vector<float> audio_copy = audio_data;
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, audio_copy.data(), audio_copy.size(),
        input_shape.data(), input_shape.size());

    // Run inference
    std::vector<const char *> input_names = {"input_values"};
    std::vector<const char *> output_names = {"logits"};

    auto inference_start = std::chrono::high_resolution_clock::now();
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
        output_names.data(), output_names.size());
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        inference_end - inference_start);
    std::cout << "Inference completed in " << inference_duration.count() << " ms" << std::endl;

    // Extract output data
    const float *output_data = outputs[0].GetTensorData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int64_t num_outputs = 1;
    for (auto dim : output_shape) {
      num_outputs *= dim;
    }

    std::vector<float> results(output_data, output_data + num_outputs);
    return results;
  }

  const ModelConfig& get_config() const { return config_; }
};

// Apply softmax to convert logits to probabilities
std::vector<float> softmax(const std::vector<float> &logits) {
  std::vector<float> probs(logits.size());
  float max_logit = *std::max_element(logits.begin(), logits.end());
  
  float sum_exp = 0.0f;
  for (size_t i = 0; i < logits.size(); i++) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum_exp += probs[i];
  }
  
  for (size_t i = 0; i < logits.size(); i++) {
    probs[i] /= sum_exp;
  }
  
  return probs;
}

int main(int argc, char *argv[]) {
  try {
    // Parse command line arguments
    std::string audio_file;
    std::string model_path_str = "models/vad/vad_model.onnx";
    std::string config_path = "models/vad/config.txt";
    bool use_openvino = false;
    bool no_cache = false;

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--help") {
        print_usage();
        return 0;
      } else if (arg == "--model" && i + 1 < argc) {
        model_path_str = argv[++i];
      } else if (arg == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (arg == "--openvino") {
        use_openvino = true;
      } else if (arg == "--no-cache") {
        no_cache = true;
      } else if (arg[0] != '-') {
        audio_file = arg;
      }
    }

    if (audio_file.empty()) {
      std::cerr << "Error: No audio file specified" << std::endl;
      print_usage();
      return 1;
    }

    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "Minimal Voice Analysis - V/A/D Emotion Recognition" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;

    // Load configuration
    std::cout << "\nLoading configuration from: " << config_path << std::endl;
    auto config = load_config(config_path);
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Sampling rate: " << config.sampling_rate << " Hz" << std::endl;
    std::cout << "  Number of labels: " << config.num_labels << std::endl;
    std::cout << "  Labels: ";
    for (size_t i = 0; i < config.label_names.size(); i++) {
      if (i > 0) std::cout << ", ";
      std::cout << config.label_names[i];
    }
    std::cout << std::endl;

    // Load audio file
    std::cout << "\nLoading audio file: " << audio_file << std::endl;
    int sample_rate;
    auto audio = load_audio(audio_file, sample_rate);
    std::cout << "Loaded audio: " << audio.size() << " samples at "
              << sample_rate << " Hz" << std::endl;

    // Resample if necessary
    if (sample_rate != config.sampling_rate) {
      std::cout << "Resampling audio from " << sample_rate << " Hz to "
                << config.sampling_rate << " Hz..." << std::endl;
      audio = resample_audio(audio, sample_rate, config.sampling_rate);
      std::cout << "Resampled audio: " << audio.size() << " samples" << std::endl;
    }

    // Calculate audio duration
    float duration_seconds = static_cast<float>(audio.size()) / config.sampling_rate;
    std::cout << "Audio duration: " << duration_seconds << " seconds" << std::endl;

    // Initialize ONNX Runtime
    std::cout << "\nInitializing ONNX Runtime..." << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MinimalVoiceAnalysis");
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
      openvino_options.device_type = "CPU_FP32";
      openvino_options.device_id = "";
      openvino_options.num_of_threads = 0;
      openvino_options.cache_dir =
          no_cache ? nullptr : model_cache.get_cache_dir().c_str();
      openvino_options.context = nullptr;
      openvino_options.enable_opencl_throttling = false;
      openvino_options.enable_dynamic_shapes = false;

      try {
        session_options.AppendExecutionProvider_OpenVINO(openvino_options);
        std::cout << "OpenVINO execution provider configured successfully" << std::endl;
        selected_device = "OpenVINO";
        
        if (!no_cache) {
          std::cout << "Model caching enabled at: " << model_cache.get_cache_dir() << std::endl;
        }
      } catch (const std::exception &e) {
        std::cout << "Failed to configure OpenVINO: " << e.what() << std::endl;
        std::cout << "Falling back to CPU execution" << std::endl;
      }
    }

    // Load model
    std::cout << "\nLoading model from: " << model_path_str << std::endl;
    std::wstring model_path(model_path_str.begin(), model_path_str.end());
    
    auto total_start = std::chrono::high_resolution_clock::now();
    VoiceAnalyzer analyzer(env, model_path, session_options, config);

    // Run analysis
    std::cout << "\nRunning voice analysis..." << std::endl;
    auto results = analyzer.analyze(audio);
    auto total_end = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);

    // Display results
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Analysis Results" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Interpret results based on number of outputs
    if (results.size() == config.num_labels) {
      // Direct regression outputs (values between 0-1 or scaled)
      std::cout << "\nEmotion Dimensions (raw scores):" << std::endl;
      for (size_t i = 0; i < results.size() && i < config.label_names.size(); i++) {
        std::cout << "  " << config.label_names[i] << ": " << results[i] << std::endl;
      }
    } else {
      // Classification outputs (logits) - apply softmax
      auto probs = softmax(results);
      std::cout << "\nEmotion Classification:" << std::endl;
      
      // Find max probability
      size_t max_idx = 0;
      float max_prob = probs[0];
      for (size_t i = 1; i < probs.size(); i++) {
        if (probs[i] > max_prob) {
          max_prob = probs[i];
          max_idx = i;
        }
      }
      
      // Display all probabilities
      for (size_t i = 0; i < probs.size() && i < config.label_names.size(); i++) {
        std::cout << "  " << config.label_names[i] << ": " 
                  << (probs[i] * 100.0f) << "%";
        if (i == max_idx) {
          std::cout << " ← Predicted";
        }
        std::cout << std::endl;
      }
    }

    // Performance summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Device: " << selected_device << std::endl;
    if (use_openvino) {
      std::cout << "Model cache used: " << (no_cache ? "No" : "Yes") << std::endl;
    }
    std::cout << "Total processing time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Audio duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "Real-time factor: " << (duration_seconds * 1000.0f / total_duration.count()) << "x" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "Use --help for usage information." << std::endl;
    return 1;
  }

  return 0;
}
