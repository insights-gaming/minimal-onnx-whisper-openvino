# Minimal Voice Analysis - V/A/D Emotion Recognition

This program performs Voice Analysis for V/A/D (Valence/Arousal/Dominance) emotion recognition using a WavLM-based Speech Emotion Recognition model.

## Model Reference

This implementation is based on the model from:
https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes

The original PyTorch implementation has been converted to ONNX format for use with ONNX Runtime and OpenVINO.

## Model Conversion

To convert the model from HuggingFace to ONNX format:

1. **Run the conversion script on a machine with internet access:**

```bash
python3 convert_vad_model.py
```

This will:
- Download the model from HuggingFace
- Convert it to ONNX format
- Save the model to `models/vad/vad_model.onnx`
- Save the configuration to `models/vad/config.txt`

2. **Copy the generated files to your target machine if needed:**

```bash
# Copy the entire models/vad directory
cp -r models/vad /path/to/target/machine/models/
```

## Building

The program is built alongside the main `minimal_whisper` executable:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This will create `minimal_voice_analysis.exe` (Windows) or `minimal_voice_analysis` (Linux).

## Usage

### Basic Usage

```bash
minimal_voice_analysis audio.wav
```

### With OpenVINO Acceleration

```bash
minimal_voice_analysis --openvino audio.wav
```

### Custom Model Path

```bash
minimal_voice_analysis --model path/to/custom_model.onnx audio.wav
```

### All Options

```bash
minimal_voice_analysis [options] <audio_file>

Options:
  --model <path>       Path to ONNX model file (default: models/vad/vad_model.onnx)
  --config <path>      Path to config file (default: models/vad/config.txt)
  --openvino           Use OpenVINO execution provider
  --no-cache           Disable OpenVINO model caching
  --help               Show help message
```

## Output

The program will output:

1. **Audio Information:**
   - Sample rate and duration
   - Resampling information if needed

2. **Analysis Results:**
   - For regression models: Raw scores for Valence, Arousal, and Dominance
   - For classification models: Probabilities for each emotion dimension

3. **Performance Metrics:**
   - Processing time
   - Real-time factor (how much faster than real-time)

### Example Output

```
============================================================
Minimal Voice Analysis - V/A/D Emotion Recognition
============================================================

Loading configuration from: models/vad/config.txt
Configuration:
  Sampling rate: 16000 Hz
  Number of labels: 3
  Labels: Valence, Arousal, Dominance

Loading audio file: audio.wav
Loaded audio: 48000 samples at 44100 Hz
Resampling audio from 44100 Hz to 16000 Hz...
Resampled audio: 17414 samples
Audio duration: 1.088 seconds

Initializing ONNX Runtime...
Loading model from: models/vad/vad_model.onnx
Model loaded successfully in 234 ms

Running voice analysis...
Inference completed in 87 ms

============================================================
Analysis Results
============================================================

Emotion Dimensions (raw scores):
  Valence: 0.672
  Arousal: 0.543
  Dominance: 0.789

============================================================
Performance Summary
============================================================
Device: CPU
Total processing time: 321 ms
Audio duration: 1.088 seconds
Real-time factor: 3.39x
```

## Model Details

The WavLM-based SER model:
- **Input:** Raw audio waveform at 16kHz sampling rate
- **Output:** Predictions for emotion dimensions (Valence, Arousal, Dominance)
- **Architecture:** Based on WavLM (Wav2Vec2-like) transformer model
- **Task:** Multi-attribute speech emotion recognition

### Emotion Dimensions

- **Valence:** Positive (happy) vs Negative (sad) emotions
- **Arousal:** High energy (excited) vs Low energy (calm) emotions
- **Dominance:** Feeling of control vs lack of control

Each dimension typically ranges from 0 to 1, where:
- 0 indicates low presence of the dimension
- 1 indicates high presence of the dimension

## Technical Implementation

The C++ implementation:
- Uses **ONNX Runtime** for model inference
- Supports **OpenVINO** acceleration for improved performance
- Includes **audio resampling** to match model requirements (16kHz)
- Uses **libsndfile** for audio file loading
- Follows the same structure as `minimal_whisper.cc`

### Key Features

1. **Efficient Audio Processing:**
   - Simple but effective resampling with anti-aliasing
   - Support for various audio formats via libsndfile

2. **Flexible Model Support:**
   - Works with both regression and classification output models
   - Configurable via external config file

3. **Performance Optimization:**
   - Optional OpenVINO acceleration
   - Model caching for faster subsequent runs
   - Real-time factor reporting

## Dependencies

- **ONNX Runtime:** For model inference
- **OpenVINO:** Optional, for accelerated inference
- **libsndfile:** For audio file I/O
- **Python 3.x** (for model conversion only):
  - transformers
  - torch
  - onnx
  - onnxruntime
  - safetensors
  - numpy

## Troubleshooting

### Model Not Found

If you see an error about the model file not being found:
1. Make sure you've run `convert_vad_model.py` first
2. Check that `models/vad/vad_model.onnx` exists
3. Use `--model` to specify a different path if needed

### Audio Format Issues

If you encounter audio loading errors:
- Ensure libsndfile supports your audio format
- Try converting to WAV format first
- Check that the file path is correct

### Performance Issues

For better performance:
- Use `--openvino` flag for OpenVINO acceleration
- Ensure model caching is enabled (don't use `--no-cache`)
- The first run will be slower due to model compilation

## License

This implementation follows the licensing of the original model and ONNX Runtime.
Please refer to the HuggingFace model page for specific license terms.
