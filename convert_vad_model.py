#!/usr/bin/env python3
"""
Script to convert the WavLM-based Speech Emotion Recognition model 
from safetensors to ONNX format for V/A/D (Valence/Arousal/Dominance) analysis.

Reference: https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes
"""

import os
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies if not already installed."""
    packages_to_check = ['transformers', 'safetensors', 'torch', 'onnx', 'onnxruntime', 'numpy']
    missing_packages = []
    
    for package in packages_to_check:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing required dependencies: {', '.join(missing_packages)}...")
        os.system(f"{sys.executable} -m pip install -q transformers safetensors torch onnx onnxruntime numpy")
        print("Dependencies installed successfully")

def convert_model_to_onnx(model_name="3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes", 
                          output_dir="models/vad"):
    """
    Convert the WavLM-based SER model to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save the ONNX model
    """
    import torch
    import numpy as np
    from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
    
    print(f"Loading model: {model_name}")
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    print("Model configuration:")
    print(f"  Sampling rate: {feature_extractor.sampling_rate}")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Number of labels: {model.config.num_labels}")
    print(f"  Label names: {model.config.id2label}")
    
    # Save configuration
    config_file = output_path / "config.txt"
    with open(config_file, "w") as f:
        f.write(f"sampling_rate={feature_extractor.sampling_rate}\n")
        f.write(f"num_labels={model.config.num_labels}\n")
        for idx, label in model.config.id2label.items():
            f.write(f"label_{idx}={label}\n")
    
    print(f"Configuration saved to {config_file}")
    
    # Create dummy input for export
    # The model expects audio input of shape (batch_size, sequence_length)
    # Use a typical audio length (e.g., 16000 samples = 1 second at 16kHz)
    batch_size = 1
    sequence_length = 16000 * 3  # 3 seconds of audio
    dummy_input = torch.randn(batch_size, sequence_length)
    
    print(f"\nExporting model to ONNX format...")
    print(f"  Input shape: {dummy_input.shape}")
    
    # Export to ONNX
    onnx_file = output_path / "vad_model.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True,
        )
    
    print(f"Model exported to {onnx_file}")
    
    # Verify the exported model
    import onnx
    onnx_model = onnx.load(str(onnx_file))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful!")
    
    # Test the ONNX model with ONNX Runtime
    print("\nTesting ONNX model with ONNX Runtime...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(onnx_file))
    
    # Run inference with dummy input
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    test_input = np.random.randn(1, sequence_length).astype(np.float32)
    outputs = session.run([output_name], {input_name: test_input})
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output logits: {outputs[0]}")
    
    print(f"\n✓ Conversion completed successfully!")
    print(f"  Model saved to: {onnx_file}")
    print(f"  Configuration saved to: {config_file}")
    
    return str(onnx_file), str(config_file)

def main():
    """Main function to run the conversion."""
    print("=" * 60)
    print("WavLM SER Model Conversion to ONNX")
    print("=" * 60)
    print("\nThis script converts the WavLM-based Speech Emotion Recognition model")
    print("from HuggingFace to ONNX format for V/A/D analysis.")
    print("\nNOTE: This script requires internet access to download the model from HuggingFace.")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Convert model
    try:
        onnx_file, config_file = convert_model_to_onnx()
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        print(f"\nTo use the model in your C++ application:")
        print(f"1. Load the ONNX model from: {onnx_file}")
        print(f"2. Read the configuration from: {config_file}")
        print(f"3. Resample audio to 16kHz")
        print(f"4. Pass audio samples as input to the model")
        print(f"5. The output will be logits for each emotion dimension")
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        print("\nIf you see a connection error, please run this script on a machine")
        print("with internet access to download and convert the model.")
        print("Then copy the generated 'models/vad' directory to this machine.")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
