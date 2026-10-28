#!/usr/bin/env bash
# GRAPHIOT launcher — the only command you need.
#   ./start.sh                 set up (first run only) and open the GUI
#   ./start.sh --port 8765     choose the port        ./start.sh --no-browser
#   ./start.sh --cuda          install the CUDA model stack (GPU server)
# System requirement: Docker (with your user in the 'docker' group). Everything
# else — Python, Node, the PLC toolchain, all packages — lives inside this folder.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"; TOOLS="$ROOT/.tools"; PORT=8765; BROWSER=1; MODEL_STACK=cpu
while [[ $# -gt 0 ]]; do case "$1" in
  --port) PORT="$2"; shift 2;; --no-browser) BROWSER=0; shift;; --cuda) MODEL_STACK=cuda; shift;;
  -h|--help) sed -n '2,8p' "$0"; exit 0;; *) echo "unknown option $1"; exit 2;; esac; done
say() { printf '\033[1;36m[graphiot]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[graphiot]\033[0m %s\n' "$*"; }
mkdir -p "$TOOLS"

# 1. Docker (the only system-level requirement) --------------------------------
if docker info >/dev/null 2>&1; then say "docker: ok"
elif sg docker -c "docker info" >/dev/null 2>&1; then say "docker: ok (via 'sg docker'; log out/in once to make it direct)"
else warn "docker is not reachable: the GUI will open but the testbed cannot start."
     warn "fix: install Docker, then  sudo usermod -aG docker \$USER  and log out/in."; fi

# 2. uv + a uv-managed Python (never the system Python) ------------------------
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$TOOLS/uv/uv" ]]; then
  say "installing uv into .tools/uv"
  curl -LsSf https://astral.sh/uv/0.11.24/install.sh | env UV_INSTALL_DIR="$TOOLS/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
UV="$(command -v uv || echo "$TOOLS/uv/uv")"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
REQ_HASH=$(cat requirements/base.txt requirements/dev.txt requirements/providers.txt pyproject.toml | sha256sum | cut -c1-16)
if [[ ! -x .venv/bin/python || "$(cat .venv/.req_hash 2>/dev/null)" != "$REQ_HASH" ]]; then
  say "setting up .venv (Python 3.12.13, pinned packages)"
  "$UV" python install 3.12.13 >/dev/null
  [[ -x .venv/bin/python ]] || "$UV" venv --python 3.12.13 --managed-python .venv >/dev/null
  VIRTUAL_ENV="$ROOT/.venv" "$UV" pip install -q -r requirements/base.txt -r requirements/dev.txt -r requirements/providers.txt -e .
  echo "$REQ_HASH" > .venv/.req_hash
fi
# Model stack: torch from the PyTorch index (PyPI's Linux wheel would pull CUDA + NVIDIA libs).
TORCH_INDEX=https://download.pytorch.org/whl/cpu; [[ $MODEL_STACK == cuda ]] && TORCH_INDEX=https://download.pytorch.org/whl/cu124
if ! .venv/bin/python -c "import torch, torch_geometric" 2>/dev/null; then
  say "installing the $MODEL_STACK model stack (torch, torch-geometric, xgboost)"
  VIRTUAL_ENV="$ROOT/.venv" "$UV" pip install -q "torch==2.14.1" --index-url "$TORCH_INDEX"
  VIRTUAL_ENV="$ROOT/.venv" "$UV" pip install -q -r "requirements/$MODEL_STACK.txt"
fi

# 3. Node 22 LTS (pinned, checksum-verified) -----------------------------------
NODE_V=v22.23.3; NODE_DIR="$TOOLS/node-$NODE_V-linux-x64"
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  say "installing Node $NODE_V into .tools"
  F="node-$NODE_V-linux-x64.tar.xz"; ( cd "$TOOLS" && curl -sSLO "https://nodejs.org/dist/$NODE_V/$F" \
    && curl -sSL "https://nodejs.org/dist/$NODE_V/SHASUMS256.txt" | grep " $F\$" | sha256sum -c - >/dev/null \
    && tar -xJf "$F" && rm -f "$F" )
fi
ln -sfn "$NODE_DIR" "$TOOLS/node"; export PATH="$TOOLS/node/bin:$PATH"

# 4. STruC++ toolchain (same compiler as OpenPLC Runtime v4) -------------------
SP=0.7.0; SPD="$ROOT/third_party/strucpp"
if [[ ! -x "$SPD/$SP/strucpp" ]]; then
  say "installing STruC++ $SP"; mkdir -p "$SPD"
  curl -sSL -o "$SPD/strucpp-linux-x64.tar.gz" "https://github.com/Autonomy-Logic/STruCpp/releases/download/v$SP/strucpp-linux-x64.tar.gz"
  tar -xzf "$SPD/strucpp-linux-x64.tar.gz" -C "$SPD" && mv "$SPD/strucpp" "$SPD/$SP" && rm -f "$SPD/strucpp-linux-x64.tar.gz"
fi
if [[ ! -f "$SPD/strucpp-$SP.tgz" ]]; then
  curl -sSL -o "$SPD/strucpp-$SP.tgz" "https://github.com/Autonomy-Logic/STruCpp/releases/download/v$SP/strucpp-$SP.tgz"
fi
sha256sum -c "$SPD/strucpp-$SP.tgz.sha256" --status 2>/dev/null || warn "STruC++ npm tarball checksum differs from the pinned one"
[[ -d plc/tools/node_modules/strucpp ]] || ( cd plc/tools && npm install --no-audit --no-fund >/dev/null )

# 5. GUI build (skipped when sources are unchanged) ----------------------------
GUI_HASH=$(cd gui && cat package.json index.html vite.config.ts $(find src -type f | sort) | sha256sum | cut -c1-16)
if [[ ! -f gui/dist/index.html || "$(cat gui/dist/.src_hash 2>/dev/null)" != "$GUI_HASH" ]]; then
  say "building the GUI"
  ( cd gui && [[ -d node_modules ]] || npm install --no-audit --no-fund >/dev/null; cd "$ROOT/gui" && npx vite build >/dev/null )
  echo "$GUI_HASH" > gui/dist/.src_hash
fi

# 6. Launch -------------------------------------------------------------------
URL="http://127.0.0.1:$PORT"
say "GUI at $URL  (Ctrl+C to stop)"
if [[ $BROWSER == 1 ]]; then ( sleep 2; xdg-open "$URL" >/dev/null 2>&1 || true ) & fi
exec .venv/bin/python -m uvicorn graphiot.server.app:app --host 127.0.0.1 --port "$PORT" --log-level warning
