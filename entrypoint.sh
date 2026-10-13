#!/bin/sh
# Starts ComfyUI with everything a user creates or installs on the /data
# volume. Extra arguments (from the Kubernetes manifest) are appended, e.g.
# --preview-method auto.
set -eu

# --base-directory /data puts custom_nodes, input, output, temp and user there.
for d in custom_nodes input output temp user user/__manager; do
  mkdir -p "/data/$d"
done
# ComfyUI doesn't create model folders in a custom models directory: make the
# usual ones so downloads have an obvious place to go.
for d in checkpoints diffusion_models text_encoders vae loras controlnet \
         clip_vision upscale_models embeddings; do
  mkdir -p "/data/models/$d"
done

# A virtualenv on the volume for custom nodes' Python packages. It sees the
# image's packages (PyTorch, ComfyUI's requirements), so it only holds what
# nodes add. The Manager installs with the Python that runs ComfyUI, i.e. this
# one. Recreated (empty) if the image's Python version changes; the Manager's
# "Install missing" / "Try fix" then reinstalls the nodes' packages.
VENV=/data/venv
PYVER=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(cat "$VENV/.python-version" 2>/dev/null)" != "$PYVER" ]; then
  echo "Creating virtualenv $VENV for Python $PYVER"
  rm -rf "$VENV"
  python -m venv --system-site-packages "$VENV"
  echo "$PYVER" > "$VENV/.python-version"
fi
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"

# Let the Manager install nodes and models: it refuses unless network_mode is
# personal_cloud when ComfyUI listens on the network. ComfyUI is reachable only
# through authentik's login (D46), so this is acceptable here.
CFG=/data/user/__manager/config.ini
if [ ! -f "$CFG" ]; then
  printf '[default]\nnetwork_mode = personal_cloud\n' > "$CFG"
elif grep -q '^network_mode' "$CFG"; then
  sed -i 's/^network_mode.*/network_mode = personal_cloud/' "$CFG"
else
  sed -i 's/^\[default\]$/[default]\nnetwork_mode = personal_cloud/' "$CFG"
fi

exec "$VENV/bin/python" /app/main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --base-directory /data \
  "$@"
