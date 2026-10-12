# ComfyUI for the homelab k3s cluster (GPU node, NVIDIA container runtime).
# The driver comes from the host at runtime; PyTorch's cu130 wheels bring the
# CUDA libraries. Everything a user creates (models, input, output, user
# settings and workflows) lives on a volume at /data, never in the image.
#
# Versions are pinned; bump them here and push to rebuild:
#   ComfyUI     v0.39.0 (commit checked below)
#   PyTorch     2.14.1 + torchvision 0.29.1, CUDA 13.0 build
#   Manager     the version ComfyUI pins in manager_requirements.txt; it runs
#               only with --enable-manager (downloads models on the server)
# Custom nodes: clone them into /app/custom_nodes below (pinned to a commit).
FROM python:3.12.15-slim-trixie

ARG COMFYUI_VERSION=v0.39.0
ARG COMFYUI_COMMIT=b0b743566f65daafc423b4fea8a2fbda94b3384a
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "$COMFYUI_VERSION" https://github.com/comfyanonymous/ComfyUI.git /app \
 && test "$(git -C /app rev-parse HEAD)" = "$COMFYUI_COMMIT" \
 && rm -rf /app/.git

WORKDIR /app

RUN pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      --index-url https://download.pytorch.org/whl/cu130 \
 && pip install -r requirements.txt \
 && pip install -r manager_requirements.txt

RUN useradd --uid 1000 --create-home comfy \
 && mkdir -p /data /app/temp \
 && chown comfy:comfy /data /app/temp

USER comfy
EXPOSE 8188

# The directories under /data are created by the entrypoint on first start.
COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
