# Ai_Reader

A lightweight, fully offline productivity tool built in Rust. This application combines local Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities into a sleek, dark-mode desktop interface.

## Features
* **Offline Dictation:** Powered by Whisper (`ggml-tiny.en.bin`) for fast, local speech-to-text transcription without requiring an internet connection.
* **Text-to-Speech:** Utilizes native Windows SAPI for clear, responsive audio playback of text.
* **Custom UI:** Built with `eframe` and `egui` for a responsive, cross-platform interface.
* **Standalone Executable:** Compiled natively for Windows with embedded custom iconography.

## Installation & Usage
1. Head to the **[Releases](../../releases)** tab on this repository.
2. Download the latest `ai_reader.exe`.
3. Download the required `ggml-tiny.en.bin` AI model and place it in the same directory as the executable.
4. Run `ai_reader.exe` to launch the application.

## Building from Source
If you want to compile the project yourself, ensure you have Rust and Cargo installed.

```bash
# Clone the repository
git clone [https://github.com/your-username/Ai_Reader.git](https://github.com/your-username/Ai_Reader.git)

# Navigate into the directory
cd Ai_Reader

# Build the highly-optimized release version
cargo build --release