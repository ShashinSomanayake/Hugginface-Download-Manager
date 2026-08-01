# HuggingFace Download Manager

## 🚀 A Production-Quality Desktop Application for Downloading AI Models

**Version:** 1.0.0
**Author:** HF Download Manager
**License:** MIT

---

## 📖 Overview

HuggingFace Download Manager is a powerful, beginner-friendly desktop application designed specifically for downloading large AI models from HuggingFace, especially **GGUF** files. It solves common problems like failed downloads, interrupted connections, and confusing file formats.

## 🚀 What Makes This App Different?

**Download large AI models (GGUF, etc.) without fear of interruption – even if:**

- 💻 **Your laptop goes to sleep** – resume automatically when you wake it up
- 📶 **Wi‑Fi disconnects** – reconnects and continues seamlessly
- 🔄 **You switch Wi‑Fi routers/networks** – the download picks up where it left off
- ⚡ **Your computer shuts down or restarts** – the queue is saved, and you can resume later
- 🌐 **You move from one network to another** (home → office → coffee shop) – no need to start over
- 🔌 **Your internet drops mid‑download** – automatic retry with smart resume

**This is not just a downloader – it's a resilient, production‑grade tool built for the real world.**

### Key Features

- ⚡ **Ultra-Fast Downloads** - Multi-connection support via aria2c
- 🔄 **Resume Support** - Never lose progress on large files
- 🎯 **GGUF Guide** - Interactive quantization comparison and RAM calculator
- 🔍 **Built-in Search** - Find models directly from the app
- 🎨 **Modern UI** - Clean, dark-themed interface (toggle between light/dark)
- 📋 **Queue Management** - Download multiple models in sequence
- 🔐 **HuggingFace Token Support** - Access private and gated models
- 🛡️ **Auto-Retry** - Automatically retry failed downloads
- 📦 **Standalone** - No Python knowledge required

---

## 🧠 How Resume Works – Deep Dive

The app uses **two complementary strategies** to guarantee resumability:

1. **aria2c (preferred)** – Creates a `.aria2` control file alongside the download that tracks exactly which parts of the file have been downloaded. Even if your computer crashes, aria2c reads that file and resumes from the exact byte offset.

2. **HTTP Range Requests (fallback)** – For streaming downloads, the app checks the size of the existing file and sends a `Range: bytes=X-` header to the server, asking for only the missing bytes.

**Real‑world scenarios that just work:**

- ✅ **Laptop closed overnight** – Open it the next morning; downloads continue from where they stopped.
- ✅ **Wi‑Fi drops for 5 minutes** – After reconnection, download resumes automatically.
- ✅ **You travel from home to office** – Connect to a new network; the download continues without user intervention.
- ✅ **Power outage mid‑download** – After reboot, launch the app; your queue is intact and downloads resume.
- ✅ **You manually pause and close the app** – The queue is saved; next launch, you can resume exactly where you paused.

---

## 📥 Installation

### Option 1: Windows (Easiest)

1. **Download** the entire project folder
2. **Double-click** `install.bat`
3. The installer will:
   - Check for Python
   - Create a virtual environment
   - Install all dependencies
   - Create a launcher (`run.bat`)
4. **Double-click** `run.bat` to start the app

### Option 2: Manual Setup (All Platforms)

```bash
# Clone or download the repository
cd hugginface-downloader

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python MAIN.py
```

### Option 3: Using pip

If you have the `MAIN.py` file, you can simply:

```bash
pip install PySide6 huggingface_hub requests tqdm
python MAIN.py
```

---

## 🎮 Quick Start Guide

### 1️⃣ **Get a HuggingFace Token** (Optional but Recommended)

- Go to [huggingface.co](https://huggingface.co)
- Create a free account (takes 30 seconds)
- Go to **Settings → Access Tokens**
- Create a new token with **Read** permission
- Copy it (starts with `hf_`)

### 2️⃣ **Download a Model**

**Method A: Paste URL**
1. Go to the **Download** tab
2. Paste a HuggingFace URL (e.g., `https://huggingface.co/TheBloke/Llama-2-7B-GGUF`)
3. Or paste a direct file link
4. Click **"START DOWNLOAD"**

**Method B: Search**
1. Go to the **Search** tab
2. Type a model name (e.g., "Llama", "Mistral")
3. Click **Search**
4. Click **"Browse"** on any result
5. Select a file and download

### 3️⃣ **Monitor Progress**

- Go to the **Queue** tab to see all downloads
- Pause, resume, or cancel downloads as needed
- See real-time speed, ETA, and progress

### 4️⃣ **Choose the Right GGUF**

Unsure which quantization to use? Go to the **GGUF Guide** tab:
- Select your model size (e.g., 7B, 13B)
- Choose a quantization
- The calculator shows estimated file size and RAM needed
- Green rows indicate recommended options

---

## 🛠️ Important Features Explained

### Resume Support

The app can resume interrupted downloads! Just:
1. Close the app mid-download
2. Reopen it
3. The download will automatically resume from where it left off

**How it works:**
- The app tracks every byte downloaded
- When resuming, it asks the server for only the missing bytes
- No need to start over from zero!

### aria2c Integration

For maximum speed, the app uses **aria2c** (a multi-connection downloader):
- Downloads files using 16 parallel connections by default
- Much faster than single-connection downloads
- Handles unstable Wi-Fi better

**Note:** If aria2c is not installed, the app will automatically download it on first run.

### HuggingFace Token Benefits

With a token, you get:
- ✅ Higher download speed limits
- ✅ Access to private/gated models
- ✅ Better reliability on HuggingFace servers
- ✅ No annoying rate-limiting

**Your token is stored locally on your computer only!**

---

## 🐛 Troubleshooting

### "Python not found" error
- Install Python 3.9 or newer from [python.org](https://www.python.org/)
- **Make sure to check "Add Python to PATH" during installation**

### Downloads keep failing
- Check your internet connection
- Add a HuggingFace token (Login tab)
- Try reducing "Connections" in the Download tab (to 8 or 4)
- Enable "Auto Retry" in Settings

### Model not showing up
- Make sure the repository exists on HuggingFace
- Some models require you to accept their license (do this on the HuggingFace website first)
- Check that you're logged in with a valid token

### "Qt binding not found"
- Run: `pip install PySide6`
- Or: `pip install PyQt6`

### Downloaded file is corrupted
- The app automatically verifies file integrity when possible
- Try re-downloading with a different method (aria2c vs HF Hub)

---

## 📚 Help & Documentation

The app includes **built-in help**! Go to the **Help** tab for articles on:
- What is GGUF?
- What is Quantization?
- Why downloads fail?
- How resume works?
- Which quant should I choose?
- What is MoE (Mixture of Experts)?
- HuggingFace Token Guide

---

## 🖥️ Requirements

- **Python:** 3.9 or newer
- **RAM:** 4 GB minimum (8+ GB recommended for large models)
- **Disk Space:** Enough for your models (models can be 5-50+ GB)
- **OS:** Windows 10+, macOS 10.15+, or modern Linux

### Python Dependencies
```
PySide6 or PyQt6    # GUI framework
huggingface_hub     # HuggingFace API
requests            # HTTP requests
tqdm                # Progress bars
```

All dependencies are automatically installed by `install.bat` or when you run `pip install -r requirements.txt`.

---

## 🔧 Advanced Settings

Go to the **Settings** tab to tweak:

| Setting | What it does |
|---------|--------------|
| **Parallel Downloads** | How many files to download at once (1-8) |
| **Connections** | Number of parallel connections per file (1-32) |
| **Max Retries** | Auto-retry count on failure (0-50) |
| **Speed Limit** | Cap download speed (0 = unlimited) |
| **Proxy** | Use an HTTP/HTTPS proxy |
| **Auto Shutdown** | Shut down PC when all downloads finish |

---

## 💡 Tips

1. **Use a token!** Even a free account gives you much better download speeds.
2. **Q4_K_M is the sweet spot** for most users (best quality/size balance).
3. **Leave at least 2-4 GB RAM free** for your operating system.
4. **Don't delete the .aria2 files** while downloading — they help with resume.
5. **Use the Queue** to download multiple models while you're away.

---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Share the app with others who need it!

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co) for their amazing platform and SDK
- [aria2](https://aria2.github.io/) for the powerful download engine
- [PySide6/PyQt6](https://www.qt.io/qt-for-python) for the GUI framework

---

## 📞 Support

- **Logs:** Check `%USERPROFILE%\.hf_download_manager\logs\` for detailed logs
- **Help Tab:** Built-in articles explain everything you need to know
- **GitHub:** Report issues or suggest features

---

## ✨ Happy Downloading!

Thank you for using HuggingFace Download Manager. We hope it makes your AI journey smoother and more enjoyable! 🚀🤗
