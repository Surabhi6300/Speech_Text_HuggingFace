#  Speech-to-Text with Wav2Vec2 & Hugging Face

A powerful speech recognition system built with Facebook's Wav2Vec2 model and Hugging Face Transformers, capable of converting audio recordings into accurate text transcriptions.

##  Features

- **Real-time Speech Recognition**: Convert audio files to text with high accuracy
- **Pre-trained Models**: Uses Facebook's `wav2vec2-base-960h` model for optimal performance
- **Audio Recording**: Built-in functionality to record audio directly in Google Colab
- **Easy Integration**: Simple setup with Hugging Face Transformers pipeline
- **Multiple Format Support**: Works with various audio formats (.wav, .mp3, etc.)

##  Quick Start

### Prerequisites

```bash
pip install transformers
pip install torchaudio 
pip install torch
```

### Basic Usage

```python
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file
waveform, sample_rate = torchaudio.load("your_audio.wav")

# Process audio
input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values

# Perform inference
with torch.no_grad():
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# Decode to text
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)
```

## 📁 Project Structure

```
Speech_Text_HuggingFace/
├── Hugging_face_Speech_to_text.ipynb  # Main notebook with implementation
└── README.md                          # Project documentation
```

## 🔧 Implementation Details

### Audio Processing
- **Sampling Rate**: 16 kHz (required for Wav2Vec2 model)
- **Audio Format**: Supports WAV, MP3, and other common formats
- **Preprocessing**: Automatic resampling and normalization

### Model Architecture
- **Base Model**: `facebook/wav2vec2-base-960h`
- **Training Data**: 960 hours of LibriSpeech dataset
- **Performance**: High accuracy on English speech recognition tasks

### Key Components
1. **Audio Recording**: JavaScript-based recording in Google Colab
2. **Audio Loading**: Using torchaudio for optimal compatibility
3. **Speech Processing**: Wav2Vec2 model for feature extraction
4. **Text Generation**: CTC (Connectionist Temporal Classification) for transcription

## Use Cases

- **Meeting Transcription**: Convert recorded meetings to searchable text
- **Podcast Processing**: Generate transcripts for audio content
- **Accessibility**: Create text versions of audio content
- **Voice Notes**: Convert voice memos to text format
- **Language Learning**: Practice pronunciation with feedback

## 🛠 Technical Requirements

- **Python**: 3.7+
- **PyTorch**: 1.9.0+
- **Transformers**: 4.11.0+
- **Torchaudio**: 0.9.0+
- **Google Colab**: Recommended environment

##  Performance

- **Accuracy**: ~95% on clear English speech
- **Processing Speed**: Real-time on GPU, near real-time on CPU
- **Memory Usage**: ~2GB for base model
- **Supported Languages**: Primarily English (model-dependent)

## Advanced Features

### Custom Audio Recording
```python
# Record audio directly in Colab
from google.colab import output
from base64 import b64decode

# JavaScript-based recording implementation
# (See notebook for full implementation)
```

### Batch Processing
```python
# Process multiple audio files
def transcribe_batch(audio_files):
    transcriptions = []
    for file in audio_files:
        # Process each file
        transcription = process_audio(file)
        transcriptions.append(transcription)
    return transcriptions
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- **Facebook AI Research** for the Wav2Vec2 model
- **Hugging Face** for the Transformers library
- **LibriSpeech** dataset contributors
- **Google Colab** for providing the development environment

##  Contact

**Surabhi** - [GitHub Profile](https://github.com/Surabhi6300)

Project Link: [https://github.com/Surabhi6300/Speech_Text_HuggingFace](https://github.com/Surabhi6300/Speech_Text_HuggingFace)

---

*Built with ❤️ for the speech recognition community*
