# ComfyUI for the homelab k3s cluster (GPU node, NVIDIA container runtime).
# The driver comes from the host at runtime; PyTorch's cu130 wheels bring the
# CUDA libraries. Everything a user creates or installs (models, input,
# output, user settings and workflows, custom nodes and their Python packages)
# lives on a volume at /data, never in the image.
#
# Versions are pinned; bump them here and push to rebuild:
#   ComfyUI     v0.39.0 (commit checked below)
#   PyTorch     2.14.1 + torchvision 0.29.1, CUDA 13.0 build
#   Manager     the version ComfyUI pins in manager_requirements.txt; it runs
#               only with --enable-manager
# Custom nodes: installed with the Manager at runtime, into /data/custom_nodes;
# their Python packages go into a virtualenv at /data/venv (see entrypoint.sh).
# Our own nodes are in nodes/ and come with the image.
FROM python:3.12.15-slim-trixie

ARG COMFYUI_VERSION=v0.39.0
ARG COMFYUI_COMMIT=b0b743566f65daafc423b4fea8a2fbda94b3384a
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# git: the Manager clones custom nodes. ffmpeg, libgl1, libglib2.0-0: what
# common custom nodes need at runtime (video nodes, OpenCV). build-essential:
# for node requirements without a prebuilt wheel. fonts-dejavu-core: the
# documentary nodes draw titles, subtitles and thumbnails with DejaVu.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 build-essential fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "$COMFYUI_VERSION" https://github.com/comfyanonymous/ComfyUI.git /app \
 && test "$(git -C /app rev-parse HEAD)" = "$COMFYUI_COMMIT" \
 && rm -rf /app/.git

WORKDIR /app

RUN pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      --index-url https://download.pytorch.org/whl/cu130 \
 && pip install -r requirements.txt \
 && pip install -r manager_requirements.txt

# Our own nodes (nodes/, reviewed in Git); the entrypoint links them into
# /data/custom_nodes. comfyui-documentary needs Kokoro TTS (weights download
# to /data/models/kokoro on first use) and spaCy's small English model, which
# Kokoro would otherwise try to download at runtime.
ARG KOKORO_VERSION=0.9.4
ARG SPACY_EN=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
RUN pip install "kokoro==$KOKORO_VERSION" "$SPACY_EN" \
 && pip check

# Chatterbox TTS (the documentary's default voice; weights download to
# /data/models/chatterbox on first use). Its own pins (torch 2.6, numpy < 2,
# diffusers 0.29, ...) would replace the image's PyTorch, so it goes in
# without dependencies, plus exactly the packages tested with it in the pod
# (2026-10-14). pip check would only report chatterbox-tts's own pins and the
# web-demo/Japanese/Chinese extras it doesn't need here (gradio, pykakasi,
# spacy-pkuseg), so it isn't run after this layer.
ARG CHATTERBOX_VERSION=0.1.7
RUN pip install --no-deps "chatterbox-tts==$CHATTERBOX_VERSION" \
 && pip install --no-deps "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu130 \
 && pip install --no-deps \
      conformer==0.3.2 diffusers==0.35.2 librosa==1.0.0 lazy-loader==0.6 \
      llvmlite==0.50.0 numba==0.68.0 decorator==5.3.1 msgpack==1.2.3 \
      pooch==1.9.0 platformdirs==4.13.0 scikit-learn==1.9.1 threadpoolctl==3.7.0 \
      soundfile==0.14.0 soxr==1.1.0 omegaconf==2.4.0 onnx==1.23.2 \
      protobuf==7.36.2 ml_dtypes==0.6.0 pyloudnorm==0.2.0 resemble-perth==1.0.1 \
      s3tokenizer==0.3.0 narwhals==2.27.1 importlib_metadata==9.0.1 zipp==4.1.1 \
      pre_commit==4.7.0 cfgv==3.5.0 identify==2.6.20 nodeenv==1.11.0 \
      virtualenv==21.14.7 distlib==0.4.3 python-discovery==1.6.2 \
 && python -c "import chatterbox.tts, torchaudio, librosa, perth, s3tokenizer"

# open_clip for BioCLIP 2, the documentary's species identifier (weights
# download to /data/models/bioclip on first use). Its other dependencies
# (regex, tqdm, PyYAML, safetensors, huggingface_hub) are already installed;
# --no-deps so it can't replace the image's PyTorch.
RUN pip install --no-deps open_clip_torch==3.3.0 timm==1.0.30 ftfy==6.3.1 wcwidth==0.9.4 \
 && python -c "import open_clip, timm, ftfy"
COPY nodes/ /opt/comfyui-nodes/

# /app/web/extensions writable: some older custom nodes (e.g.
# ComfyUI-Custom-Scripts) copy their web files there on every start and fail
# to load if they can't. It is rebuilt by them at start, so it needn't persist.
RUN useradd --uid 1000 --create-home comfy \
 && mkdir -p /data /app/web/extensions \
 && chown comfy:comfy /data /app/web /app/web/extensions

USER comfy
EXPOSE 8188

# The directories under /data are created by the entrypoint on first start.
COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
