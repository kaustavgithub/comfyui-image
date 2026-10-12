# comfyui-image

A pinned [ComfyUI](https://github.com/comfyanonymous/ComfyUI) container image
for an NVIDIA GPU node (driver from the host, CUDA 13.0 PyTorch in the image).

- Built by GitHub Actions on every push to `main`:
  `ghcr.io/kaustavgithub/comfyui-image:sha-<commit>` and `:latest`.
- Versions are pinned in the `Dockerfile` (ComfyUI release + commit, PyTorch).
- All user data goes to a volume at `/data` (`models/`, `input/`, `output/`,
  `user/`); the entrypoint creates the folders.
- No ComfyUI-Manager: custom nodes are added in the `Dockerfile`, pinned to a
  commit, so every change is reviewed in Git.
- ComfyUI has no authentication. Never expose port 8188 directly; put it
  behind an authenticating proxy.

Run locally:

```sh
docker run --rm --gpus all -p 8188:8188 -v comfy-data:/data \
  ghcr.io/kaustavgithub/comfyui-image:latest --preview-method auto
```
