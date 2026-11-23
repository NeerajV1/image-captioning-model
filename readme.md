cat << 'EOF' > README.md
# Multimodal Image Captioning (CLIP + LSTM + Bahdanau Attention)

An encoder-decoder image captioning model trained from scratch on the Flickr8k dataset.

## Architecture
- **Vision Encoder:** Frozen CLIP image encoder extracting spatial visual feature embeddings.
- **Decoder:** Autoregressive LSTM network generating token-by-token captions.
- **Attention Mechanism:** Additive Bahdanau attention aligning decoder hidden states with spatial image regions at each generation step.

## Status
Core architecture, training pipeline, and attention mechanism implemented and trained on Flickr8k. Inference script and demo cleanup in progress.
EOF