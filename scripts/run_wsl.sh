#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FEMWEB_VENV:-$HOME/.venvs/femweb}"
PORT="${FEMWEB_PORT:-8765}"

. "$VENV/bin/activate"
cd "$ROOT"

export PYTHONUTF8=1
export FEMWEB_DEVICE="${FEMWEB_DEVICE:-auto}"
export FEMWEB_THREADS="${FEMWEB_THREADS:-auto}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-}},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
if [ -n "${FEMWEB_CA_BUNDLE:-}" ]; then
  export REQUESTS_CA_BUNDLE="$FEMWEB_CA_BUNDLE"
  export SSL_CERT_FILE="$FEMWEB_CA_BUNDLE"
fi

URL="http://localhost:$PORT"
(
  sleep 2
  if command -v wslview >/dev/null 2>&1; then
    wslview "$URL" >/dev/null 2>&1 || true
  elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$URL" >/dev/null 2>&1 || true
  fi
) &

echo "Windows 브라우저에서 $URL 을 여세요 (자동으로 열리지 않으면 직접 입력)"
exec python -m femweb --host 127.0.0.1 --port "$PORT" --no-browser
