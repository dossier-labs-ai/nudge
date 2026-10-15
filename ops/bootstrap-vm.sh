#!/bin/bash
# Run as root on a dedicated Debian VM. No secrets or generated scripts.
set -euo pipefail
if [ "$#" -lt 8 ] || [ "$#" -gt 9 ] || { [ "$#" -eq 9 ] && [ "$9" != --resume ]; }; then
  echo 'usage: bootstrap-vm.sh REPO_ROOT PREPARED_JSON OUTPUT_GCS_URI GATEWAY_IMAGE RUN_ROOT BASETEN_MODEL SECRET_PROJECT SECRET_REF [--resume]' >&2
  exit 2
fi
repo_root="$1"
prepared_json="$2"
output_uri="$3"
gateway_image="$4"
run_root="$5"
baseten_model="$6"
secret_project="$7"
secret_ref="$8"
apt-get update
apt-get install -y python3-venv docker.io
systemctl enable --now docker
python3 -m venv /opt/rune-bench-venv
/opt/rune-bench-venv/bin/pip install "$repo_root[operations]"
docker pull "$gateway_image"
exec /opt/rune-bench-venv/bin/python -m nudge.routerbench_vm run \
  --root="$run_root" \
  --prepared="$prepared_json" \
  --gateway-config="$repo_root/gateway/routerbench.toml" \
  --gateway-image="$gateway_image" \
  --output-uri="$output_uri" \
  --model="$baseten_model" --environment=eval \
  --secret-project="$secret_project" --secret="$secret_ref" \
  --max-hours=24 --shutdown "${@:9}"
