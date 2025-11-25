#!/bin/bash

echo "Starting Ollama model downloads..."

# List of models to download (you can customize this list)
#"llama3.2:1b"
#"llama3.2:3b"
#"mistral:7b"
MODELS=(
    "qwen2.5:0.5b"
)

for model in "${MODELS[@]}"; do
    echo "Downloading model: $model"
    ollama pull $model

    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded: $model"
    else
        echo "❌ Failed to download: $model"
    fi
done

echo "Model download process completed!"
echo "Available models:"
ollama list