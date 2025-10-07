# Quick Start Guide: V/A/D Voice Analysis

This guide will help you get started with the V/A/D (Valence/Arousal/Dominance) voice analysis program in just a few steps.

## What is V/A/D Analysis?

V/A/D analysis measures three dimensions of emotion in speech:
- **Valence:** How positive or negative the emotion is (happy vs sad)
- **Arousal:** The energy level of the emotion (excited vs calm)
- **Dominance:** The sense of control in the emotion (dominant vs submissive)

## Quick Start (3 Steps)

### 1. Convert the Model (One-time setup)

**Requirements:** Internet access + Python 3

```bash
python3 convert_vad_model.py
```

This downloads and converts the WavLM model from HuggingFace to ONNX format.

**What you get:**
- `models/vad/vad_model.onnx` (the ONNX model)
- `models/vad/config.txt` (model configuration)

**Note:** If you don't have internet access, run this on another machine and copy the `models/vad/` folder.

---

### 2. Build the Program

**Requirements:** Windows + CMake + Visual Studio

```cmd
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**What you get:**
- `build/Release/minimal_voice_analysis.exe`

---

### 3. Analyze Audio

```cmd
cd build\Release
minimal_voice_analysis.exe your_audio.wav
```

**Example output:**
```
Emotion Dimensions (raw scores):
  Valence: 0.723    ← Positive emotion
  Arousal: 0.612    ← Moderate-high energy
  Dominance: 0.834  ← High control
```

---

## Understanding the Results

### Score Ranges (0.0 to 1.0)

**Valence:**
- 0.0-0.3: Negative (sad, angry)
- 0.3-0.7: Neutral
- 0.7-1.0: Positive (happy, joyful)

**Arousal:**
- 0.0-0.3: Low energy (calm, sleepy)
- 0.3-0.7: Moderate energy
- 0.7-1.0: High energy (excited, alert)

**Dominance:**
- 0.0-0.3: Submissive (influenced)
- 0.3-0.7: Moderate control
- 0.7-1.0: Dominant (in control)

### Example Interpretations

| Valence | Arousal | Dominance | Emotion |
|---------|---------|-----------|---------|
| 0.85    | 0.90    | 0.80      | **Joy/Excitement** |
| 0.20    | 0.85    | 0.75      | **Anger** |
| 0.15    | 0.80    | 0.25      | **Fear** |
| 0.25    | 0.20    | 0.30      | **Sadness** |
| 0.75    | 0.30    | 0.70      | **Contentment** |

---

## Performance Tips

### Speed up with OpenVINO:
```cmd
minimal_voice_analysis.exe --openvino audio.wav
```

**Benefits:**
- 2-3x faster processing
- First run is slow (model compilation), subsequent runs are fast
- Automatic model caching

### Supported Audio Formats:
- WAV, FLAC, OGG, MP3
- Any sample rate (automatically resampled to 16kHz)
- Mono or stereo (stereo converted to mono)

---

## Troubleshooting

### "Failed to load model"
→ Make sure you ran `python3 convert_vad_model.py` first

### "OpenVINO failed"
→ No problem! It will fall back to CPU

### Slow first run
→ Normal! OpenVINO compiles the model on first run (~10-30s)
→ Subsequent runs are much faster (cached)

---

## Command Line Options

```
minimal_voice_analysis [options] <audio_file>

Options:
  --model <path>    Custom model path
  --config <path>   Custom config path
  --openvino        Use OpenVINO acceleration
  --no-cache        Disable model caching
  --help            Show help
```

---

## Complete Documentation

For detailed information, see:
- **VAD_README.md** - Full documentation
- **USAGE_EXAMPLE.md** - Step-by-step examples
- **convert_vad_model.py** - Model conversion script

---

## Technical Details

**Model:** WavLM-based Speech Emotion Recognition  
**Source:** https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes  
**Framework:** ONNX Runtime with OpenVINO support  
**Input:** 16kHz audio (auto-resampled)  
**Output:** 3 scores (Valence, Arousal, Dominance)

**Implementation:**
- C++20
- Similar architecture to `minimal_whisper.cc`
- Uses libsndfile for audio I/O
- Simple but effective resampling

---

## Example Workflow

```cmd
# 1. Convert model (one time)
python3 convert_vad_model.py

# 2. Build (one time)
mkdir build && cd build
cmake .. && cmake --build . --config Release

# 3. Run analysis (repeat as needed)
cd Release
minimal_voice_analysis.exe --openvino sample1.wav
minimal_voice_analysis.exe --openvino sample2.wav
minimal_voice_analysis.exe --openvino sample3.wav
```

---

## Next Steps

1. **Try different audio files** - Test with various emotional speech samples
2. **Integrate into your app** - Use the code as reference for your C++ applications
3. **Optimize performance** - Experiment with OpenVINO settings
4. **Custom models** - Train your own models using the same architecture

---

Happy analyzing! 🎤🔊📊
