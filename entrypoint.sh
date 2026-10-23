#!/bin/bash

# ComfyUI log file
COMFY_LOG="/ComfyUI/comfyui.log"

echo "=== Starting ComfyUI in the background ==="
cd /ComfyUI

# Link models from /runpod-volume/models to /ComfyUI/models if volume exists
if [ -d "/runpod-volume/models" ]; then
    echo "Found /runpod-volume/models, linking subfolders to /ComfyUI/models..."
    for dir in /runpod-volume/models/*; do
        if [ -d "$dir" ]; then
            base_dir=$(basename "$dir")
            echo "Linking model directory: $base_dir"
            mkdir -p "/ComfyUI/models/$base_dir"
            ln -sfn "$dir"/* "/ComfyUI/models/$base_dir/" 2>/dev/null || true
        fi
    done
fi

# ComfyUI ko --extra-model-paths-config ke sath run karein

python main.py --listen 127.0.0.1 --port 8188 --extra-model-paths-config /ComfyUI/extra_model_paths.yaml --use-sage-attention > "$COMFY_LOG" 2>&1 &
COMFY_PID=$!

echo "Waiting for ComfyUI to be ready (PID: $COMFY_PID)..."
max_wait=180
wait_count=0

while [ $wait_count -lt $max_wait ]; do
    # Agar ComfyUI crash ho jaye toh turant log dikha kar exit karein
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "❌ Error: ComfyUI process crashed unexpectedly! Full logs:"
        cat "$COMFY_LOG"
        exit 1
    fi

    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/')" > /dev/null 2>&1; then
        echo "✅ ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "❌ Error: ComfyUI failed to start within $max_wait seconds. Full logs:"
    cat "$COMFY_LOG"
    exit 1
fi

# Start the handler in the foreground with unbuffered output (-u)
echo "=== Starting RunPod Handler ==="
exec python -u /handler.py