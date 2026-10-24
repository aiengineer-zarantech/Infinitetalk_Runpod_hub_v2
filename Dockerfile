# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa requests

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# Pin transformers to 4.x. In transformers 5.x the Wav2Vec2 encoder no longer
# returns hidden_states (they are captured by a decorator on the parent forward),
# and WanVideoWrapper's custom Wav2Vec2Model overrides that forward, so
# MultiTalkWav2VecEmbeds (node 194) gets hidden_states=None and crashes.
# transformers 4.x needs huggingface_hub<1.0, and diffusers>=0.40 needs
# huggingface_hub>=1.x (resolve_revision), which breaks the whole WanVideoWrapper
# import. diffusers 0.39.0 is the last release that supports huggingface_hub 0.x.
RUN pip install "transformers==4.57.6" "diffusers==0.39.0" && \
    python -c "import transformers; v = transformers.__version__; print('transformers', v); assert v.startswith('4.'), v" && \
    python -c "import diffusers.configuration_utils, diffusers, huggingface_hub; print('diffusers', diffusers.__version__, 'huggingface_hub', huggingface_hub.__version__)"

COPY . .

# extra_model_paths ko ComfyUI folder mein copy karein
RUN cp /extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml

# Entrypoint aur setup_models se CRLF hatao aur executable banayein
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 755 /entrypoint.sh
RUN sed -i 's/\r$//' /setup_models.py && chmod 755 /setup_models.py

CMD ["/entrypoint.sh"]