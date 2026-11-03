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

# v27: ComfyUI core gets --use-sage-attention only if the sageattn self-test in gpu_check.py passed (the installed
# kernels are Blackwell-only) and ATTENTION_MODE is not sdpa. No self-test result: unchanged (flag kept).
GPU_INFO="${WORKER_GPU_INFO:-/tmp/worker_gpu.json}"
SAGE_ARGS=(--use-sage-attention)
SAGE_STATUS=$("$PYTHON_BIN" -c "import json, sys; print(json.load(open(sys.argv[1])).get('sageattn') or '')" "$GPU_INFO" 2>/dev/null)
ATTN_MODE=$(printf '%s' "${ATTENTION_MODE:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
if [ "$ATTN_MODE" = "sdpa" ]; then
    SAGE_ARGS=()
    echo "ATTENTION_MODE=sdpa: ComfyUI starts without --use-sage-attention"
elif [ -n "$SAGE_STATUS" ] && [ "$SAGE_STATUS" != "ok" ]; then
    SAGE_ARGS=()
    echo "sageattn self-test did not pass: ComfyUI starts without --use-sage-attention, jobs use sdpa"
fi

# v27: torch.compile caches on the network volume, so a new worker reuses earlier compiles. Only when the volume
# is mounted and writable; COMPILE_CACHE=0 keeps the container defaults. Values already set in the env win.
COMPILE_CACHE_VOLUME="${COMPILE_CACHE_VOLUME:-/runpod-volume}"
COMPILE_CACHE_ROOT="$COMPILE_CACHE_VOLUME/cache/compile"
if [ "${COMPILE_CACHE:-1}" = "0" ]; then
    echo "[COMPILE] cache: COMPILE_CACHE=0, compile caches stay in the container (lost when the worker stops)"
elif case "$COMPILE_CACHE_VOLUME" in *huggingface-cache*) true ;; *) false ;; esac; then
    # RunPod Cached Models: RunPod-managed, possibly read-only and shared between endpoints. Never write there.
    echo "[COMPILE] cache: $COMPILE_CACHE_VOLUME is inside the RunPod HF cache, compile caches stay in the container (not persistent)"
elif [ -d "$COMPILE_CACHE_VOLUME" ] && [ -w "$COMPILE_CACHE_VOLUME" ] \
        && mkdir -p "$COMPILE_CACHE_ROOT/inductor" "$COMPILE_CACHE_ROOT/triton" 2>/dev/null; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$COMPILE_CACHE_ROOT/inductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$COMPILE_CACHE_ROOT/triton}"
    export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
    export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
    echo "[COMPILE] cache: TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
else
    echo "[COMPILE] cache: $COMPILE_CACHE_VOLUME not mounted or not writable, compile caches stay in the container"
fi

# perf/5090-speedup: where the models come from (setup_models.py does the work and logs [TIMING] model setup)
MODEL_SOURCE_NOW=$(printf '%s' "${MODEL_SOURCE:-volume}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
VOLUME_CHECK="${VOLUME_ROOT_CHECK:-/runpod-volume}"
VOLUME_STATE="not mounted"
if [ -d "$VOLUME_CHECK" ]; then
    VOLUME_STATE=$([ -w "$VOLUME_CHECK" ] && echo "mounted, writable" || echo "mounted, read-only")
fi
echo "[MODELS] MODEL_SOURCE=$MODEL_SOURCE_NOW, HF_CACHE_REPO=${HF_CACHE_REPO:-unset}, $VOLUME_CHECK $VOLUME_STATE"

# perf/5090-speedup: ComfyUI-Manager makes network calls at startup (registry / update checks) and no workflow
# node needs it. COMFYUI_MANAGER=0 (default) renames it to *.disabled, which ComfyUI skips (nodes.py:
# load_custom_nodes); COMFYUI_MANAGER=1 restores it.
MANAGER_DIR="$COMFY_DIR/custom_nodes/ComfyUI-Manager"
if [ "${COMFYUI_MANAGER:-0}" = "1" ]; then
    if [ -d "$MANAGER_DIR.disabled" ] && [ ! -e "$MANAGER_DIR" ]; then
        mv "$MANAGER_DIR.disabled" "$MANAGER_DIR" && echo "[COMFY] ComfyUI-Manager enabled (COMFYUI_MANAGER=1)"
    fi
elif [ -d "$MANAGER_DIR" ]; then
    rm -rf "$MANAGER_DIR.disabled"
    mv "$MANAGER_DIR" "$MANAGER_DIR.disabled" \
        && echo "[COMFY] ComfyUI-Manager disabled (COMFYUI_MANAGER=0): no startup network calls"
fi

echo "=== Model Setup & Discovery (background) ==="
"$PYTHON3_BIN" -u "$WORKER_ROOT/setup_models.py" &
COPY_PID=$!

echo "=== Starting ComfyUI in the background ==="
cd "$COMFY_DIR" || { echo "❌ $COMFY_DIR not found"; kill "$COPY_PID" 2>/dev/null; exit 1; }

# Test-only allocator switch (default: unset = ComfyUI's cudaMallocAsync, exactly as before).
# expandable_segments only works with torch's native allocator; ComfyUI appends backend:cudaMallocAsync
# to PYTORCH_CUDA_ALLOC_CONF unless it is started with --disable-cuda-malloc.
COMFY_ALLOC_ARGS=()
# MODEL_LOCK (default on): expandable segments against fragmentation, so the resident model and the per-job
# tensors fit without an OOM (an OOM makes ComfyUI unload every model). ALLOC_MODE=cuda_malloc keeps ComfyUI's
# default cudaMallocAsync allocator.
if [ -z "${ALLOC_MODE:-}" ] && [ "${MODEL_LOCK:-1}" != "0" ]; then
    ALLOC_MODE=native_expandable
    echo "[MODEL_LOCK] ALLOC_MODE unset: using native_expandable"
fi
if [ "${ALLOC_MODE:-}" = "cuda_malloc" ]; then
    echo "ALLOC_MODE=cuda_malloc: ComfyUI default allocator (cudaMallocAsync)"
elif [ "${ALLOC_MODE:-}" = "native_expandable" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    COMFY_ALLOC_ARGS=(--disable-cuda-malloc)
    echo "ALLOC_MODE=native_expandable: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF, ComfyUI --disable-cuda-malloc (test only)"
elif [ -n "${ALLOC_MODE:-}" ]; then
    echo "⚠️ unknown ALLOC_MODE='$ALLOC_MODE' ignored (supported: native_expandable)"
fi

# ComfyUI ko --extra-model-paths-config ke sath run karein (GPU only, no CPU fallback flags).
# Output goes to comfyui.log (the handler reads its tail on errors) and to the worker log as [comfy].
PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py --listen 127.0.0.1 --port "$COMFY_PORT" --extra-model-paths-config "$COMFY_DIR/extra_model_paths.yaml" "${SAGE_ARGS[@]}" "${COMFY_ALLOC_ARGS[@]}" \
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
# deploy/pod: start_pod.sh sets POD_MODE=1. POD_API=pod (default) = authenticated job API (pod/pod_api.py);
# POD_API=runpod_sim = the RunPod SDK dev simulator (no auth, runs the job inside POST /status: debug only).
if [ "${POD_MODE:-0}" = "1" ]; then
    if [ "${POD_API:-pod}" = "runpod_sim" ]; then
        echo "=== Starting RunPod SDK local API (debug, NO AUTH) on ${POD_API_HOST:-0.0.0.0}:${POD_API_PORT:-8000} ==="
        exec "$PYTHON_BIN" -u "$WORKER_ROOT/handler.py" --rp_serve_api \
            --rp_api_host "${POD_API_HOST:-0.0.0.0}" --rp_api_port "${POD_API_PORT:-8000}"
    fi
    echo "=== Starting pod job API on ${POD_API_HOST:-0.0.0.0}:${POD_API_PORT:-8000} ==="
    exec "$PYTHON_BIN" -u /pod/pod_api.py
fi

echo "=== Starting RunPod Handler ==="
exec "$PYTHON_BIN" -u "$WORKER_ROOT/handler.py"
