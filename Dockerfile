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
# for node requirements without a prebuilt wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 build-essential \
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
COPY nodes/ /opt/comfyui-nodes/

RUN useradd --uid 1000 --create-home comfy \
 && mkdir -p /data \
 && chown comfy:comfy /data

USER comfy
EXPOSE 8188

# The directories under /data are created by the entrypoint on first start.
COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
