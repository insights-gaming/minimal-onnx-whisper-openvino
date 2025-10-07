# Usage Example: Voice Analysis with V/A/D Model

This document provides step-by-step instructions for using the V/A/D voice analysis program.

## Prerequisites

### On the Machine for Model Conversion (needs internet):
- Python 3.x
- pip

### On the Machine for Running the Analysis:
- Windows OS (this project is configured for Windows)
- CMake 3.15 or higher
- Visual Studio 2019 or higher with C++ support

## Step 1: Convert the Model to ONNX Format

**Run this on a machine with internet access:**

```bash
# Navigate to the project directory
cd /path/to/minimal-onnx-whisper-openvino

# Run the conversion script
python3 convert_vad_model.py
```

The script will:
1. Install required Python packages (transformers, torch, onnx, etc.)
2. Download the WavLM model from HuggingFace (3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes)
3. Convert it to ONNX format
4. Save the model to `models/vad/vad_model.onnx`
5. Save the configuration to `models/vad/config.txt`

**Expected Output:**
```
============================================================
WavLM SER Model Conversion to ONNX
============================================================

This script converts the WavLM-based Speech Emotion Recognition model
from HuggingFace to ONNX format for V/A/D analysis.

NOTE: This script requires internet access to download the model from HuggingFace.
============================================================
Loading model: 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes
Model configuration:
  Sampling rate: 16000
  Model type: wavlm
  Number of labels: 3
  Label names: {0: 'Valence', 1: 'Arousal', 2: 'Dominance'}
Configuration saved to models/vad/config.txt

Exporting model to ONNX format...
  Input shape: torch.Size([1, 48000])
Model exported to models/vad/vad_model.onnx
ONNX model validation successful!

Testing ONNX model with ONNX Runtime...
  Input shape: (1, 48000)
  Output shape: (1, 3)
  Output logits: [[0.123 -0.456 0.789]]

✓ Conversion completed successfully!
  Model saved to: models/vad/vad_model.onnx
  Configuration saved to: models/vad/config.txt

============================================================
Conversion completed successfully!
============================================================

To use the model in your C++ application:
1. Load the ONNX model from: models/vad/vad_model.onnx
2. Read the configuration from: models/vad/config.txt
3. Resample audio to 16kHz
4. Pass audio samples as input to the model
5. The output will be logits for each emotion dimension
```

## Step 2: Build the C++ Program

**On Windows:**

```cmd
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project (Release configuration)
cmake --build . --config Release

# The executable will be in: build/Release/minimal_voice_analysis.exe
```

## Step 3: Prepare Audio Files

The program supports various audio formats (WAV, FLAC, OGG, etc.) through libsndfile.

**Example audio files:**
- Any recording of speech with emotion
- Duration: Any length (model will process the entire file)
- Sample rate: Any (will be resampled to 16kHz automatically)
- Channels: Mono or stereo (stereo will be converted to mono)

## Step 4: Run the Voice Analysis

### Basic Usage (CPU):

```cmd
# Navigate to the Release directory
cd build\Release

# Run analysis on an audio file
minimal_voice_analysis.exe ..\..\test_audio.wav
```

### With OpenVINO Acceleration:

```cmd
# Use OpenVINO for faster inference
minimal_voice_analysis.exe --openvino ..\..\test_audio.wav
```

### With Custom Model:

```cmd
# Use a custom model and config
minimal_voice_analysis.exe --model custom_model.onnx --config custom_config.txt audio.wav
```

## Step 5: Interpret the Results

### Example Output:

```
============================================================
Minimal Voice Analysis - V/A/D Emotion Recognition
============================================================

Loading configuration from: ..\..\models\vad\config.txt
Configuration:
  Sampling rate: 16000 Hz
  Number of labels: 3
  Labels: Valence, Arousal, Dominance

Loading audio file: ..\..\test_audio.wav
Loaded audio: 220500 samples at 44100 Hz
Resampling audio from 44100 Hz to 16000 Hz...
Resampled audio: 80000 samples
Audio duration: 5.000 seconds

Initializing ONNX Runtime...
Configuring OpenVINO execution provider...
OpenVINO execution provider configured successfully
Model caching enabled at: openvino_cache
Loading model from: ..\..\models\vad\vad_model.onnx
Model loaded successfully in 1234 ms

Running voice analysis...
Inference completed in 156 ms

============================================================
Analysis Results
============================================================

Emotion Dimensions (raw scores):
  Valence: 0.723    ← Positive emotion (happy/pleasant)
  Arousal: 0.612    ← Moderate-high energy level
  Dominance: 0.834  ← High sense of control

============================================================
Performance Summary
============================================================
Device: OpenVINO
Model cache used: Yes
Total processing time: 1390 ms
Audio duration: 5.000 seconds
Real-time factor: 3.60x
```

### Understanding the Scores:

**Valence (Positive vs Negative):**
- **0.0 - 0.3:** Negative emotion (sad, angry, fearful)
- **0.3 - 0.7:** Neutral emotion
- **0.7 - 1.0:** Positive emotion (happy, joyful, pleased)

**Arousal (Energy Level):**
- **0.0 - 0.3:** Low arousal (calm, relaxed, sleepy)
- **0.3 - 0.7:** Moderate arousal
- **0.7 - 1.0:** High arousal (excited, alert, active)

**Dominance (Control):**
- **0.0 - 0.3:** Low dominance (submissive, influenced)
- **0.3 - 0.7:** Moderate dominance
- **0.7 - 1.0:** High dominance (in control, dominant)

### Emotion Mapping Examples:

| Valence | Arousal | Dominance | Likely Emotion |
|---------|---------|-----------|----------------|
| High    | High    | High      | Happy, Excited |
| High    | Low     | High      | Content, Relaxed |
| Low     | High    | High      | Angry, Frustrated |
| Low     | High    | Low       | Fearful, Anxious |
| Low     | Low     | Low       | Sad, Depressed |

## Troubleshooting

### Issue: "Failed to open audio file"
**Solution:** Check that the file path is correct and the audio format is supported.

### Issue: "Failed to load model"
**Solution:** 
1. Ensure you ran `convert_vad_model.py` successfully
2. Check that `models/vad/vad_model.onnx` exists
3. Verify the model path in the command line

### Issue: "OpenVINO configuration failed"
**Solution:** 
- OpenVINO will fall back to CPU if configuration fails
- Check that OpenVINO is installed correctly
- The program will still work with CPU execution

### Issue: Very slow first run
**Solution:** 
- The first run compiles the model for OpenVINO (takes ~10-30 seconds)
- Subsequent runs will be much faster due to caching
- Use `--no-cache` to disable caching if needed

## Advanced Usage

### Batch Processing Multiple Files:

```cmd
# Windows batch script
@echo off
for %%f in (*.wav) do (
    echo Processing: %%f
    minimal_voice_analysis.exe --openvino "%%f" >> results.txt
)
```

### Processing with Different Sampling Rates:

The program automatically resamples audio to 16kHz. You can use audio files at any sample rate:
- 8kHz (telephone quality)
- 16kHz (wideband)
- 44.1kHz (CD quality)
- 48kHz (professional audio)

### Real-time Processing:

For real-time processing, the program needs to process faster than real-time (RTF > 1.0):
- **CPU only:** RTF ~0.5-2.0x (depending on CPU)
- **OpenVINO:** RTF ~2.0-5.0x (depending on hardware)

If RTF > 1.0, the program can process audio in real-time.

## Performance Tips

1. **Use OpenVINO:** Add `--openvino` flag for 2-3x speedup
2. **Enable caching:** Don't use `--no-cache` (caching is on by default)
3. **Use shorter audio files:** Process in chunks if needed
4. **First run:** The first run will be slower due to model compilation

## Next Steps

- **Integrate into applications:** Use the code as a reference for your own C++ applications
- **Experiment with different audio:** Try various emotional speech samples
- **Optimize for your hardware:** Test with different OpenVINO devices (CPU, GPU, NPU)
- **Custom models:** Train your own models and convert them using the same process
