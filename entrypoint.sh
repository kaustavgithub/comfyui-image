#!/bin/sh
# Starts ComfyUI with all user data on the /data volume. Extra arguments (from
# the Kubernetes manifest) are appended, e.g. --preview-method auto.
set -eu

for d in input output user; do
  mkdir -p "/data/$d"
done
# ComfyUI doesn't create model folders in a custom models directory: make the
# usual ones so downloads have an obvious place to go.
for d in checkpoints diffusion_models text_encoders vae loras controlnet \
         clip_vision upscale_models embeddings; do
  mkdir -p "/data/models/$d"
done

exec python /app/main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --models-directory /data/models \
  --input-directory /data/input \
  --output-directory /data/output \
  --user-directory /data/user \
  "$@"
