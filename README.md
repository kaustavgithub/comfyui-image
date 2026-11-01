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
- Our own node packs and workflows are **not** in the image: they are in
  [comfyui-workflows](https://github.com/kaustavgithub/comfyui-workflows),
  which the homelab's deployment fetches at a pinned commit and links into
  `/data/custom_nodes` at every start, so a node or workflow change needs no
  image build. The image has the Python packages they need: Kokoro (+ spaCy's
  small English model), Chatterbox and open_clip (BioCLIP 2) for
  `comfyui-documentary`. A new package means an image build (or the Manager,
  into `/data/venv`).
- The Manager runs with `use_uv = False` (set by the entrypoint): uv ignores
  the image's packages and would reinstall them into the venv.
- ComfyUI has no authentication, and anyone who can reach it can install
  code with the Manager. Never expose port 8188 directly; put it behind an
  authenticating proxy.

Run locally:

```sh
docker run --rm --gpus all -p 8188:8188 -v comfy-data:/data \
  ghcr.io/kaustavgithub/comfyui-image:latest --preview-method auto
```
