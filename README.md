# comfyui-image

A pinned [ComfyUI](https://github.com/comfyanonymous/ComfyUI) container image
for an NVIDIA GPU node (driver from the host, CUDA 13.0 PyTorch in the image).

- Built by GitHub Actions on every push to `main`:
  `ghcr.io/kaustavgithub/comfyui-image:sha-<commit>` and `:latest`.
- Versions are pinned in the `Dockerfile` (ComfyUI release + commit, PyTorch).
- Everything a user creates or installs goes to a volume at `/data`
  (`--base-directory /data`): `models/`, `input/`, `output/`, `user/`,
  `custom_nodes/`, and `venv/`, a virtualenv on top of the image's packages
  for the custom nodes' Python packages. The entrypoint creates them.
- ComfyUI-Manager is installed (the version ComfyUI pins) and runs with
  `--enable-manager`. The entrypoint sets its `network_mode` to
  `personal_cloud`, so it can install custom nodes and models; they persist on
  the volume across restarts and image updates. ComfyUI itself is updated by
  bumping the `Dockerfile`, not from the Manager.
- ComfyUI has no authentication, and anyone who can reach it can install
  code with the Manager. Never expose port 8188 directly; put it behind an
  authenticating proxy.

Run locally:

```sh
docker run --rm --gpus all -p 8188:8188 -v comfy-data:/data \
  ghcr.io/kaustavgithub/comfyui-image:latest --preview-method auto
```
