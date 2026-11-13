#!/usr/bin/env bash
# Install the CoSpaDi environment from uv.lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/' >&2
  exit 1
fi
exec uv sync --locked --extra eval "$@"
