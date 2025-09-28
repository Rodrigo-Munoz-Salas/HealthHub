# Quantization Options

## Current Status
- TinyLlama quantized model: 1.1GB (still too large for GitHub)
- MiniLM model: 87MB (within GitHub limits)

## Further Quantization Options
1. **INT4 quantization**: Could reduce size by ~50%
2. **INT2 quantization**: Could reduce size by ~75% (but quality loss)
3. **Dynamic quantization**: Runtime quantization
4. **Pruning + quantization**: Remove unnecessary weights

## GitHub-Friendly Solutions
1. Keep MiniLM (87MB) in repo
2. Host TinyLlama on Hugging Face Hub
3. Use GitHub Releases for TinyLlama
4. Further quantize TinyLlama to <100MB
