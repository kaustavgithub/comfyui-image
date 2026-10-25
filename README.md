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
  YouTube video (16:9) or an Instagram reel (9:16, at most 60 s). Template
  "Documentary from photos" in ComfyUI's template browser.
- Species: BioCLIP 2 (open_clip, `species.py`) names a bird's or mammal's
  species when it is sure, among the world's birds and mammals (GBIF; no
  location is used), weighted by how common each one is; a name given on the
  photo, or a species named in its note, wins. Unsure: the story says only "a
  duck" and uses no facts. The facts are each species' Wikipedia intro,
  downloaded once with the list into `/data/models/bioclip`; a narration's
  number that neither they, the note nor the camera settings have is
  rewritten or dropped.
- Stories open with a hook line. Mood "behind the lens": the photographer's
  own story of each shot, from its camera settings and time. Your own voice
  (my_voice) speaks calmly in every mood, close to the recording.
- Offline: the documentary nodes send no photo, note or story anywhere but the
  LLM gate. Their models load from the volume without asking Hugging Face for
  updates, and the facts are on the volume too: once each file has been
  downloaded (on its first use), a workflow runs without the internet. The
  only requests out are those one-time downloads (model weights; the
  species list and facts, asked by scientific names).
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
