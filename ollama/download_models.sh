#!/bin/bash

models=(
    "mxbai-embed-large"
    "llama3.2:3b"
)

for model in "${models[@]}"; do
    docker exec -it ollama ollama pull "$model"
done