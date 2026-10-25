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
- Our own nodes live in `nodes/` and come with the image; the entrypoint
  links them into `/data/custom_nodes`. `comfyui-documentary`: photos →
  per photo, a vision model names the subject and a text model (both through
  the homelab's LLM gate, OpenAI API: `DOC_LLM_URL`, `DOC_LLM_TOKEN`)
  writes the narration → Chatterbox or Kokoro text-to-speech → music → video
  with an intro collage, transitions, subtitles and a follow card, plus a
  thumbnail and a social post (caption, hashtags, key phrases), for a
  YouTube video (16:9) or an Instagram reel (9:16, at most 40 s). Template
  "Documentary from photos" in ComfyUI's template browser.
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
