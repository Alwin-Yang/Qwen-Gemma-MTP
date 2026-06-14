# How to Run MTP Models: Multi-Token Prediction Guide

The Unsloth documentation provides a comprehensive guide on how to run MTP models.

See this [guide](https://unsloth.ai/docs/models/mtp) for more details.

> [!NOTE] This guide is targeted for running models on one single NVIDIA 5090 GPU.

---
## Run with Unsloth Studio

### 1. Install Unsloth
Run in your terminal:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```


### 2. Launch Unsloth
MacOS, Linux, WSL and Windows:
```bash
unsloth studio -H 127.0.0.1 -p 8888
```

Then open http://127.0.0.1:8888 (or your specific URL) in your browser.


### 3. Search and download your desired model
On first launch you will need to create a password to secure your account and sign in again later. 
>[!NOTE] Gemma 4 MTP is automatically enabled in Unsloth. You only need to download the regular original Gemma 4 GGUF. 

| Gemma 4 | Qwen3.6 MTP |
|:-:|:-:|
| ![Search and download Gemma 4 in Unsloth Studio](images/gemma4.png) | ![Search and download Qwen3.6 MTP in Unsloth Studio](images/qwen3.6.png) |


### 4. Run your MTP model
Inference, MTP and speculative decoding settings should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit speculative decoding, the context length, chat template and other settings in the right side bar.



---
## Run with Llama.cpp




