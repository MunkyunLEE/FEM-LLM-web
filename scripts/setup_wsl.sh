#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FEMWEB_VENV:-$HOME/.venvs/femweb}"
PY="${FEMWEB_PYTHON:-python3}"

APT_PKGS="python3-venv python3-dev build-essential calculix-ccx libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 libfontconfig1"

if [ "${1:-}" = "--apt" ]; then
  sudo -E apt-get update
  sudo -E apt-get install -y $APT_PKGS
fi

ver=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
"$PY" -c 'import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)' || { echo "Python 3.10 이상이 필요합니다 (현재 $ver)"; exit 1; }

if [ -n "${FEMWEB_CA_BUNDLE:-}" ]; then
  export PIP_CERT="$FEMWEB_CA_BUNDLE"
  export REQUESTS_CA_BUNDLE="$FEMWEB_CA_BUNDLE"
  export SSL_CERT_FILE="$FEMWEB_CA_BUNDLE"
fi

[ -d "$VENV" ] || "$PY" -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install -U pip wheel
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install -r "$ROOT/requirements-solid.txt"

python - <<'EOF'
import shutil
import torch
print("python  ", __import__("sys").version.split()[0])
print("torch   ", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
try:
    import ipctk
    print("ipctk   ", ipctk.__version__)
except Exception as e:
    print("ipctk    없음:", e)
try:
    import gmsh
    gmsh.initialize(); gmsh.finalize()
    print("gmsh     OK")
except Exception as e:
    print("gmsh     실패:", e, "-> ./scripts/setup_wsl.sh --apt 로 시스템 라이브러리를 설치하세요")
print("ccx     ", shutil.which("ccx") or "없음 -> sudo apt install calculix-ccx")
EOF

echo
echo "설치 완료. 실행: $ROOT/scripts/run_wsl.sh"
