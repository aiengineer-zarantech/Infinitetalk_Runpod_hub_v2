#!/bin/bash
# Startup order: GPU guard -> model copy and ComfyUI start in parallel -> wait for both
# -> ComfyUI device check -> handler. Any failure exits non-zero so RunPod restarts the worker.

# Path overrides exist only so this script can be smoke-tested outside the image
WORKER_ROOT="${WORKER_ROOT:-}"
COMFY_DIR="${COMFY_DIR:-/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_START_TIMEOUT="${COMFY_START_TIMEOUT:-180}"

# ComfyUI log file
COMFY_LOG="$COMFY_DIR/comfyui.log"

now() { date +%s.%N; }
T0=$(now)

echo "=== GPU Guard ==="
"$PYTHON_BIN" -u "$WORKER_ROOT/worker_guard/gpu_check.py" || { echo "❌ GPU guard failed, exiting so RunPod restarts this worker"; exit 1; }
T_GPU=$(now)

echo "=== Model Setup & Discovery (background) ==="
"$PYTHON3_BIN" -u "$WORKER_ROOT/setup_models.py" &
COPY_PID=$!

echo "=== Starting ComfyUI in the background ==="
cd "$COMFY_DIR" || { echo "❌ $COMFY_DIR not found"; kill "$COPY_PID" 2>/dev/null; exit 1; }

# Test-only allocator switch (default: unset = ComfyUI's cudaMallocAsync, exactly as before).
# expandable_segments only works with torch's native allocator; ComfyUI appends backend:cudaMallocAsync
# to PYTORCH_CUDA_ALLOC_CONF unless it is started with --disable-cuda-malloc.
COMFY_ALLOC_ARGS=()
if [ "${ALLOC_MODE:-}" = "native_expandable" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    COMFY_ALLOC_ARGS=(--disable-cuda-malloc)
    echo "ALLOC_MODE=native_expandable: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF, ComfyUI --disable-cuda-malloc (test only)"
elif [ -n "${ALLOC_MODE:-}" ]; then
    echo "⚠️ unknown ALLOC_MODE='$ALLOC_MODE' ignored (supported: native_expandable)"
fi

# ComfyUI ko --extra-model-paths-config ke sath run karein (GPU only, no CPU fallback flags).
# Output goes to comfyui.log (the handler reads its tail on errors) and to the worker log as [comfy].
PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py --listen 127.0.0.1 --port "$COMFY_PORT" --extra-model-paths-config "$COMFY_DIR/extra_model_paths.yaml" --use-sage-attention "${COMFY_ALLOC_ARGS[@]}" \
    > >(tee "$COMFY_LOG" | sed -u 's/^/[comfy] /') 2>&1 &
COMFY_PID=$!

fail() {
    echo "❌ Error: $1"
    kill "$COPY_PID" "$COMFY_PID" 2>/dev/null
    exit 1
}

echo "Waiting for model copy (PID: $COPY_PID) and ComfyUI (PID: $COMFY_PID)..."
COPY_DONE=""
COMFY_READY=""
T_COPY=""
T_COMFY=""
WAIT_START=$(date +%s)
LAST_STATUS=0

while [ -z "$COPY_DONE" ] || [ -z "$COMFY_READY" ]; do
    # Agar ComfyUI crash ho jaye toh turant exit karein (its output is above as [comfy] lines)
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        fail "ComfyUI process crashed unexpectedly! See the [comfy] lines above (full log: $COMFY_LOG)"
    fi

    if [ -z "$COPY_DONE" ] && ! kill -0 "$COPY_PID" 2>/dev/null; then
        wait "$COPY_PID"
        COPY_RC=$?
        [ "$COPY_RC" -eq 0 ] || fail "setup_models.py failed (exit code $COPY_RC)!"
        COPY_DONE=1
        T_COPY=$(now)
        echo "✅ Model copy finished"
    fi

    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [ -z "$COMFY_READY" ]; then
        if "$PYTHON_BIN" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$COMFY_PORT/', timeout=5)" > /dev/null 2>&1; then
            COMFY_READY=1
            T_COMFY=$(now)
            echo "✅ ComfyUI is ready!"
        elif [ "$ELAPSED" -ge "$COMFY_START_TIMEOUT" ]; then
            fail "ComfyUI failed to start within $COMFY_START_TIMEOUT seconds. See the [comfy] lines above (full log: $COMFY_LOG)"
        fi
    fi

    if [ -z "$COPY_DONE" ] || [ -z "$COMFY_READY" ]; then
        if [ $(( ELAPSED - LAST_STATUS )) -ge 10 ]; then
            LAST_STATUS=$ELAPSED
            echo "Waiting... (${ELAPSED}s) model copy: $([ -n "$COPY_DONE" ] && echo done || echo running), ComfyUI: $([ -n "$COMFY_READY" ] && echo ready || echo starting)"
        fi
        sleep 1
    fi
done

"$PYTHON_BIN" -u "$WORKER_ROOT/worker_guard/startup_check.py" --t0 "$T0" --gpu-done "$T_GPU" --copy-done "$T_COPY" --comfy-ready "$T_COMFY" \
    || fail "ComfyUI device check failed"

# Start the handler in the foreground with unbuffered output (-u)
echo "=== Starting RunPod Handler ==="
exec "$PYTHON_BIN" -u "$WORKER_ROOT/handler.py"
