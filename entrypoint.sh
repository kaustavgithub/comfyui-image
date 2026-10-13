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

# Our own nodes come with the image (/opt/comfyui-nodes): link each into
# custom_nodes, replacing an older link.
for n in /opt/comfyui-nodes/*/; do
  n=${n%/}
  ln -sfn "$n" "/data/custom_nodes/$(basename "$n")"
done

# Manager settings, enforced on every start:
# - network_mode = personal_cloud: it refuses to install nodes and models
#   otherwise when ComfyUI listens on the network. ComfyUI is reachable only
#   through authentik's login (D46), so this is acceptable here.
# - use_uv = False: uv ignores the image's packages (it doesn't honour the
#   venv's system-site-packages) and would reinstall them into /data/venv;
#   pip sees them as installed.
CFG=/data/user/__manager/config.ini
[ -f "$CFG" ] || printf '[default]\n' > "$CFG"
set_cfg() {
  if grep -q "^$1 *=" "$CFG"; then
    sed -i "s/^$1 *=.*/$1 = $2/" "$CFG"
  else
    sed -i "s/^\\[default\\]\$/[default]\\n$1 = $2/" "$CFG"
  fi
}
set_cfg network_mode personal_cloud
set_cfg use_uv False

exec "$VENV/bin/python" /app/main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --base-directory /data \
  "$@"
