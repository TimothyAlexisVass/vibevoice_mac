#!/usr/bin/env bash
set -Eeuo pipefail

#==============================================================
# VibeVoice macOS Apple Silicon setup & runner (fully-contained)
#==============================================================
# Requirements met:
# - Creates local project folders (./.venv, ./_cache, ./models, ./VibeVoice, ./outputs)
# - No sudo / no global installs (unless --allow-brew for ffmpeg)
# - Idempotent; safe to re-run; offline-friendly reuse
# - Uses an existing local model directory (no HF downloads)
# - Provides --demo (Gradio) and --infer (CLI-ish) paths
# - Forces all Hugging Face caches into ./_cache
# - Avoids CUDA/FlashAttention; uses PyTorch MPS if available
#
# Usage examples:
#   bash setup_vibevoice_mac.sh --demo
#   bash setup_vibevoice_mac.sh --infer
#   bash setup_vibevoice_mac.sh --model-path ./models/VibeVoice-7B --demo --share
#   bash setup_vibevoice_mac.sh --clean --force
#
# Flags:
#   --model-path <path>   Local model directory (default: ./models/VibeVoice-7B)
#   --demo                Launch Gradio demo on port 7860
#   --share               Add --share to Gradio (optional)
#   --infer               Run a simple CLI inference example from text file
#   --allow-brew          If ffmpeg missing, permit Homebrew install (no sudo)
#   --clean               Remove venv, caches, repo, models, outputs (prompt unless --force)
#   --force               Use with --clean to skip prompt
#   -h | --help           Show this help
#
# Notes:
# - Tested targets: macOS (Darwin) + arm64 (Apple Silicon) only; exits otherwise
# - Python: requires >= 3.10 (uses system python3); will not install Python
# - All data stays under the project directory (this repo)
#==============================================================

### Pretty logging
if [[ -t 1 ]]; then
  # shellcheck disable=SC2034
  RED="$(printf '\033[31m')" GREEN="$(printf '\033[32m')" YELLOW="$(printf '\033[33m')"
  # shellcheck disable=SC2034
  BLUE="$(printf '\033[34m')" BOLD="$(printf '\033[1m')" RESET="$(printf '\033[0m')"
else
  RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
fi
info()  { printf "%b[INFO]%b %s\n"  "$BLUE" "$RESET" "$*"; }
warn()  { printf "%b[WARN]%b %s\n"  "$YELLOW" "$RESET" "$*"; }
error() { printf "%b[ERROR]%b %s\n" "$RED" "$RESET" "$*" >&2; }
success(){ printf "%b[DONE]%b %s\n" "$GREEN" "$RESET" "$*"; }

### Trap for friendly error messages
on_error() {
  error "An unexpected error occurred. See messages above."
  warn  "Common fixes:
  - Ensure stable internet for first run (pip, git, model download).
  - Check Python >= 3.10: run 'python3 --version'.
  - If MPS unavailable, the script will fall back to CPU (slower).
  - If SSL/cert errors occur, try: 'export PIP_DISABLE_PIP_VERSION_CHECK=1' and re-run, or ensure your macOS certificates are up to date.
  - If port 7860 is in use (for --demo), close that app or change PORT env var for this run, e.g.: PORT=7861 --demo
  - For low disk space: free up space or move PROJECT_DIR (see below)."
}
trap on_error ERR

### Defaults & globals
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/WhoPaidItAll/VibeVoice"
REPO_DIR="${PROJECT_DIR}/VibeVoice"
VENV_DIR="${PROJECT_DIR}/.venv"
CACHE_DIR="${PROJECT_DIR}/_cache"
MODELS_DIR="${PROJECT_DIR}/models"
TOOLS_DIR="${PROJECT_DIR}/tools"
FFMPEG_DIR="${TOOLS_DIR}/ffmpeg"
OUTPUTS_DIR="${PROJECT_DIR}/outputs"
TMP_DIR="${PROJECT_DIR}/_tmp"

MODEL_PATH_DEFAULT="${MODELS_DIR}/VibeVoice-7B"
MODEL_PATH="${MODEL_PATH_DEFAULT}"

DO_DEMO=0
DO_SHARE=0
DO_INFER=0
ALLOW_BREW=0
DO_CLEAN=0
FORCE=0

### Help text
usage() {
  sed -n '1,70p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

### Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) shift; MODEL_PATH="${1:-}"; [[ -z "${MODEL_PATH}" ]] && { error "--model-path requires a value"; exit 2; } ;;
    --model) shift; MODEL_PATH="${1:-}"; [[ -z "${MODEL_PATH}" ]] && { error "--model requires a value"; exit 2; } ;;
    --demo) DO_DEMO=1 ;;
    --share) DO_SHARE=1 ;;
    --infer) DO_INFER=1 ;;
    --allow-brew) ALLOW_BREW=1 ;;
    --clean) DO_CLEAN=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage ;;
    *) error "Unknown flag: $1"; usage ;;
  esac
  shift || true
done

### Clean mode
if [[ "$DO_CLEAN" -eq 1 ]]; then
  if [[ "$FORCE" -eq 0 ]]; then
    read -r -p "This will permanently delete runtime data under ${PROJECT_DIR} (venv, caches, models, outputs, tools, and the VibeVoice checkout). Proceed? [y/N] " ans
    case "${ans:-N}" in
      y|Y|yes|YES) : ;;
      *) info "Aborted."; exit 0 ;;
    esac
  fi
  to_remove=(
    "${VENV_DIR}"
    "${CACHE_DIR}"
    "${MODELS_DIR}"
    "${TOOLS_DIR}"
    "${OUTPUTS_DIR}"
    "${TMP_DIR}"
    "${REPO_DIR}"
    "${PROJECT_DIR}/.vv_bootstrap.py"
    "${PROJECT_DIR}/demo_text.txt"
  )
  removed_any=0
  for p in "${to_remove[@]}"; do
    if [[ -e "$p" ]]; then
      info "Removing $p ..."
      rm -rf "$p"
      removed_any=1
    fi
  done
  if [[ "$removed_any" -eq 1 ]]; then
    success "Cleaned runtime directories."
  else
    info "Nothing to clean; no runtime directories found."
  fi
  exit 0
fi

### Platform checks
OS="$(uname -s || true)"
ARCH="$(uname -m || true)"
if [[ "${OS}" != "Darwin" ]]; then
  error "This script supports macOS only. Detected: ${OS}."
  exit 1
fi
if [[ "${ARCH}" != "arm64" ]]; then
  error "This script targets Apple Silicon (arm64) only. Detected: ${ARCH}."
  exit 1
fi

### Python checks (use system python3; do not install)
if ! command -v python3 >/dev/null 2>&1; then
  error "python3 not found. Please install Python 3.10+ via python.org or Homebrew, then re-run."
  exit 1
fi
PY_VER_STR="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PY_MAJ="$(python3 -c 'import sys; print(sys.version_info[0])')"
PY_MIN="$(python3 -c 'import sys; print(sys.version_info[1])')"
if (( PY_MAJ < 3 || (PY_MAJ == 3 && PY_MIN < 10) )); then
  error "Python ${PY_VER_STR} detected. Python >= 3.10 is required. Please upgrade and re-run."
  exit 1
fi
info "Using Python ${PY_VER_STR} (system)."

### Prepare directories
mkdir -p "${PROJECT_DIR}" "${CACHE_DIR}" "${MODELS_DIR}" "${TOOLS_DIR}" "${FFMPEG_DIR}" "${OUTPUTS_DIR}" "${TMP_DIR}"

### Constrain all caches/logs to project dir (privacy & isolation)
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export NUMBA_CACHE_DIR="${CACHE_DIR}/numba"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TMPDIR="${TMP_DIR}"
export VIBEVOICE_OUTPUTS="${OUTPUTS_DIR}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
# Avoid CUDA/FlashAttention assumptions; force safe attention math
export USE_FLASH_ATTENTION=0
export FLASH_ATTENTION_SKIP=1
export ATTN_BACKEND=math
export PYTORCH_ENABLE_MPS_FALLBACK=1

### Show disk space (troubleshoot)
if command -v df >/dev/null 2>&1; then
  FREE_KB="$(df -Pk "${PROJECT_DIR}" | awk 'NR==2{print $4}')"
  if [[ -n "${FREE_KB}" ]]; then
    FREE_GB="$(awk "BEGIN {printf \"%.1f\", ${FREE_KB}/1024/1024}")"
    info "Approx free space at project volume: ${FREE_GB} GB"
    if awk "BEGIN {exit !(${FREE_KB} < 5242880)}"; then  # < 5 GB
      warn "Low disk space (<5GB). Model downloads may fail."
    fi
  fi
fi

### Create/Reuse venv
if [[ ! -d "${VENV_DIR}" ]]; then
  info "Creating virtual environment at ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
else
  info "Reusing existing virtual environment at ${VENV_DIR} ..."
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PATH="${VENV_DIR}/bin:${FFMPEG_DIR}:$PATH"

### Upgrade base tooling (local only)
python -m pip install --upgrade --no-input pip setuptools wheel

### Install Python deps (no CUDA, MPS-capable torch from PyPI)
info "Installing Python dependencies (local venv) ..."
python - <<'PY'
import sys, subprocess
pkgs = [
    # Core
    "torch",                       # macOS wheels include MPS
    "transformers",
    "accelerate",
    "huggingface_hub[cli]",
    "soundfile",
    "numpy",
    "scipy",
    "gradio",
    # Utilities
    "imageio-ffmpeg",             # to fetch a portable ffmpeg binary if needed
    "packaging",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", *pkgs])
PY

### Quick MPS sanity check (warn and continue if unavailable)
python - <<'PY' || true
import torch, sys
mps_ok = torch.backends.mps.is_available()
print(f"[MPS] Available: {mps_ok}")
try:
    if mps_ok:
        x = torch.ones((1024,1024), device="mps")
        y = torch.mm(x, x)
        print("[MPS] Basic tensor op succeeded.")
    else:
        print("[MPS] Not available; will proceed on CPU (slower).")
except Exception as e:
    print(f"[MPS] Warning: MPS test raised: {e}\nProceeding on CPU.", file=sys.stderr)
PY

### Ensure ffmpeg available (prefer portable local; avoid global installs)
need_ffmpeg=0
if ! command -v ffmpeg >/dev/null 2>&1; then
  need_ffmpeg=1
fi

if [[ "${need_ffmpeg}" -eq 1 ]]; then
  info "ffmpeg not found on PATH. Obtaining a portable binary into ${FFMPEG_DIR} ..."
  FFMPEG_BIN="${FFMPEG_DIR}/ffmpeg"
  if [[ -x "${FFMPEG_BIN}" ]]; then
    info "Using previously downloaded ffmpeg at ${FFMPEG_BIN}"
  else
    if [[ "${ALLOW_BREW}" -eq 1 ]] && command -v brew >/dev/null 2>&1; then
      info "--allow-brew enabled; attempting Homebrew install (no sudo) ..."
      brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
      if command -v ffmpeg >/dev/null 2>&1; then
        info "Homebrew ffmpeg installed."
      fi
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
      info "Falling back to Python 'imageio-ffmpeg' portable binary (stored inside project caches) ..."
      export IMAGEIO_USERDIR="${CACHE_DIR}/imageio"
      python - <<'PY'
import os, shutil
from pathlib import Path
import imageio_ffmpeg
exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"[ffmpeg] imageio-ffmpeg provided: {exe}")
ffmpeg_dir = Path(os.environ["FFMPEG_DIR"])
ffmpeg_dir.mkdir(parents=True, exist_ok=True)
target = ffmpeg_dir / "ffmpeg"
# Copy (not symlink) to satisfy "portable binary inside ./tools/ffmpeg/"
shutil.copy2(exe, target)
os.chmod(target, 0o755)
print(f"[ffmpeg] Copied to: {target}")
PY
      if [[ ! -x "${FFMPEG_BIN}" ]]; then
        error "Failed to obtain a portable ffmpeg binary."
        warn  "Tip: re-run with --allow-brew (requires Homebrew installed) or install ffmpeg yourself, then re-run."
        exit 1
      fi
    fi
  fi
else
  info "System ffmpeg found: $(command -v ffmpeg)"
fi
export PATH="${FFMPEG_DIR}:$PATH"

### Clone or update VibeVoice repo (safe offline)
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  info "Cloning VibeVoice repository into ${REPO_DIR} ..."
  if ! git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"; then
    warn "Clone failed (possibly offline). If the repo was previously present, continuing with existing contents if any."
    mkdir -p "${REPO_DIR}"
  fi
else
  info "Updating VibeVoice repository (fetch/pull) ..."
  if ! (git -C "${REPO_DIR}" fetch --depth 1 && git -C "${REPO_DIR}" pull --ff-only); then
    warn "Update failed (possibly offline). Reusing existing repo at ${REPO_DIR}."
  fi
fi

### Install the repo in editable mode (local venv)
if [[ -f "${REPO_DIR}/pyproject.toml" || -f "${REPO_DIR}/setup.py" ]]; then
  info "Installing VibeVoice in editable mode ..."
  (cd "${REPO_DIR}" && python -m pip install --no-input -e .) || warn "Editable install failed; continuing (some demos may still work if deps are already satisfied)."
else
  warn "Repo seems incomplete (no pyproject.toml/setup.py). Continuing anyway."
fi

### Resolve local model directory (no network / no HF)
resolve_model_path() {
  local input="$1"
  local resolved=""
  local status=0

  if resolved="$(MODEL_PATH_IN="$input" MODELS_DIR="${MODELS_DIR}" python - <<'PY'
import os, sys

base = os.environ.get("MODEL_PATH_IN", "")
models_dir = os.environ.get("MODELS_DIR", "")

def has_any_files(d):
    try:
        return any(os.scandir(d))
    except OSError:
        return False

def has_large_safetensors(d, min_bytes=10 * 1024 * 1024):
    if not os.path.isdir(d):
        return False
    for name in os.listdir(d):
        if name.endswith(".safetensors"):
            p = os.path.join(d, name)
            try:
                if os.path.getsize(p) > min_bytes:
                    return True
            except OSError:
                pass
    return False

def find_valid(root, max_depth=3):
    root = os.path.realpath(root)
    for curr, dirs, _files in os.walk(root):
        depth = curr[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        if has_large_safetensors(curr):
            return os.path.realpath(curr)
    return None

if not os.path.isdir(base):
    sys.exit(2)
if not has_any_files(base):
    sys.exit(3)
if has_large_safetensors(base):
    print(os.path.realpath(base))
    sys.exit(0)

found = find_valid(base, 3)
if not found and models_dir and os.path.isdir(models_dir):
    found = find_valid(models_dir, 3)

if found:
    print(found)
    sys.exit(0)

sys.exit(4)
PY
)"; then
    status=0
  else
    status=$?
  fi

  case "${status}" in
    0)
      if [[ -z "${resolved}" ]]; then
        error "Model path resolution failed for: ${input}"
        exit 1
      fi
      if [[ "${resolved}" != "${input}" ]]; then
        warn "Model path looked incomplete; using detected model dir: ${resolved}"
      fi
      MODEL_PATH="${resolved}"
      ;;
    2)
      error "Local model path not found: ${input}"
      warn "Place your model files there or pass --model-path /path/to/model"
      exit 1
      ;;
    3)
      error "Local model path is empty: ${input}"
      warn "Ensure the model files are fully downloaded."
      exit 1
      ;;
    4)
      error "Local model path has no real .safetensors weights (likely Git LFS pointers): ${input}"
      warn "Point --model-path at the folder that contains the large model shards."
      exit 1
      ;;
    *)
      error "Model path resolution failed for: ${input}"
      exit 1
      ;;
  esac
}

resolve_model_path "${MODEL_PATH}"
success "Using local model at: ${MODEL_PATH}"

# --- VibeVoice mac bootstrap (auto-patches at runtime) ---
vv_write_bootstrap() {
  # writes a small launcher that:
  #  - forces attn_implementation='sdpa'
  #  - stubs HF's CUDA allocator warmup
  #  - forces device_map to mps/cpu (never cuda)
  #  - forwards all CLI args to the original gradio_demo.py
  cat > ".vv_bootstrap.py" <<'PY'
import os, sys, runpy

# Force no CUDA; keep MPS fallback
os.environ.setdefault('ACCELERATE_DISABLE_CUDA', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Patch Transformers: SDPA, no CUDA warmup, and never dispatch to CUDA
try:
    from transformers import modeling_utils as _mu
    import torch

    # 1) Kill CUDA caching allocator warmup entirely (safe on CPU/MPS)
    try:
        _mu.caching_allocator_warmup = lambda *a, **k: None
    except Exception:
        pass

    # 2) Wrap from_pretrained to force safe kwargs & device placement
    _orig = _mu.PreTrainedModel.from_pretrained
    def _patched_from_pretrained(cls, path, *args, **kwargs):
        # neutralize FlashAttention and enforce SDPA
        kwargs.pop('use_flash_attention_2', None)
        if kwargs.get('attn_implementation') != 'sdpa':
            kwargs['attn_implementation'] = 'sdpa'

        # do NOT pass this kwarg through (custom classes may reject it)
        kwargs.pop('caching_allocator_warmup', None)

        # Force away from CUDA no matter what
        want_mps = torch.backends.mps.is_available()
        target_dev = 'mps' if want_mps else 'cpu'

        dm = kwargs.get('device_map', 'auto')
        if (
            dm == 'auto'
            or (isinstance(dm, str) and 'cuda' in dm)
            or (isinstance(dm, dict) and any('cuda' in str(v) for v in dm.values()))
        ):
            kwargs['device_map'] = target_dev

        # MPS is more stable with float16 than bf16
        if want_mps:
            td = kwargs.get('torch_dtype')
            if td is None or td == torch.bfloat16:
                kwargs['torch_dtype'] = torch.float16

        return _orig.__func__(cls, path, *args, **kwargs)

    _mu.PreTrainedModel.from_pretrained = classmethod(_patched_from_pretrained)
except Exception as e:
    print(f"[WARN] bootstrap: could not patch transformers: {e}", file=sys.stderr)

# Hand off to the original demo, preserving CLI args
demo = os.path.join(os.path.dirname(__file__), 'VibeVoice', 'demo', 'gradio_demo.py')
sys.argv = [demo] + sys.argv[1:]
runpy.run_path(demo, run_name='__main__')
PY
}
# --- end bootstrap ---



### Demo & inference runners
PORT="${PORT:-7860}"

ensure_port_free() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"${port}" -sTCP:LISTEN -n >/dev/null 2>&1; then
    error "Port ${port} is already in use. Close the process using it or set PORT=<free_port> when re-running."
    exit 1
  fi
}

run_gradio_demo() {
  ensure_port_free "${PORT}"
  info "Launching Gradio demo on http://127.0.0.1:${PORT} ..."
  local demo_script="${REPO_DIR}/demo/gradio_demo.py"
  if [[ ! -f "${demo_script}" ]]; then
    error "Gradio demo script not found at ${demo_script}. The repo layout may have changed."
    exit 1
  fi
  local -a cmd
  cmd=(
    python .vv_bootstrap.py
    --model_path "${MODEL_PATH}"
    --port "${PORT}"
    --device "$(python -c 'import torch; print("mps" if torch.backends.mps.is_available() else "cpu")')"
  )
  if [[ "${DO_SHARE}" -eq 1 ]]; then cmd+=(--share); fi
  # Use the local model files to avoid network; launch through our bootstrap
  (
    cd "${PROJECT_DIR}" && \
    vv_write_bootstrap; \
    ACCELERATE_DISABLE_CUDA=1 CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 \
      exec "${cmd[@]}"
  ) || {
    error "Gradio demo exited unexpectedly."
    exit 1
  }
}

run_cli_infer() {
  info "Running CLI-style inference example ..."
  local text_file=""
  if [[ -f "${REPO_DIR}/demo/text_examples/1p_abs.txt" ]]; then
    text_file="${REPO_DIR}/demo/text_examples/1p_abs.txt"
  else
    text_file="${PROJECT_DIR}/demo_text.txt"
    printf "This is a short sample for VibeVoice on macOS.\n" > "${text_file}"
    info "Created example text: ${text_file}"
  fi

  # Try known candidate scripts in the repo; if none found, fall back to a lightweight local runner.
  declare -a candidates=(
    "${REPO_DIR}/demo/cli_infer.py"
    "${REPO_DIR}/demo/cli_tts.py"
    "${REPO_DIR}/demo/infer_cli.py"
    "${REPO_DIR}/demo/tts_cli.py"
    "${REPO_DIR}/examples/cli_infer.py"
  )
  local runner=""
  for c in "${candidates[@]}"; do
    [[ -f "$c" ]] && { runner="$c"; break; }
  done

  mkdir -p "${OUTPUTS_DIR}"
  local out_wav="${OUTPUTS_DIR}/sample_out.wav"

  if [[ -n "${runner}" ]]; then
    info "Using repo CLI: ${runner}"
    # Heuristic args; many repos use --model_path and --text_file/--output
    if ! python "${runner}" --model_path "${MODEL_PATH}" --text_file "${text_file}" --output "${out_wav}" 2>/dev/null; then
      warn "Repo CLI invocation failed (args may differ). Falling back to a minimal local runner."
    else
      success "WAV written to: ${out_wav}"
      return 0
    fi
  fi

  # Minimal local runner: try to use the installed package API if available; otherwise, use Transformers Auto classes.
  python - <<PY || {
import sys, os, soundfile as sf
from pathlib import Path

model_path = os.environ.get("MODEL_PATH")
text_path = os.environ.get("TEXT_FILE")
out_wav = os.environ.get("OUT_WAV")

text = Path(text_path).read_text().strip()
if not text:
    text = "Hello from VibeVoice."

# Try VibeVoice API if present
try:
    import vibevoice  # type: ignore
    # Generic pseudo-API; adapt gracefully if not found
    if hasattr(vibevoice, "load_model") and hasattr(vibevoice, "tts"):
        model = vibevoice.load_model(model_path)
        audio = vibevoice.tts(model, text=text, speaker="female", sample_rate=24000)
        sf.write(out_wav, audio, 24000)
        print(f"[OK] Wrote {out_wav} via vibevoice API.")
        sys.exit(0)
except Exception as e:
    print(f"[Fallback] vibevoice API not usable: {e}")

# Try Transformers (if model compatible)
try:
    from transformers import AutoProcessor, AutoModel
    import torch, numpy as np
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    prompt = text
    # Many TTS repos expose generate or inference methods with remote code
    if hasattr(model, "generate"):
        inputs = processor(text=prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        # Heuristic: retrieve audio from output dict/tuple
        wav = None
        if isinstance(out, dict):
            wav = out.get("audio") or out.get("waveform")
        elif isinstance(out, (list, tuple)) and out:
            wav = out[0]
        if wav is None:
            raise RuntimeError("Model.generate() did not return audio.")
        arr = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(out_wav, arr, 24000)
        print(f"[OK] Wrote {out_wav} via transformers.generate()")
        sys.exit(0)
    else:
        raise RuntimeError("Model has no .generate; cannot proceed.")
except Exception as e:
    print(f"[ERROR] Local fallback inference failed: {e}")
    sys.exit(2)
PY
    error "CLI inference failed. The repo may have changed its API. Try --demo for a working Gradio UI."
    exit 1
  }
  success "WAV written to: ${out_wav}"
}

### Final tips (performance)
warn "If inference is slow on MPS/CPU, try a smaller local model via:
  --model-path /path/to/smaller/model
and/or split long text into shorter chunks."

### Execute requested action
if [[ "${DO_DEMO}" -eq 1 && "${DO_INFER}" -eq 1 ]]; then
  warn "--demo and --infer both specified; running --infer first, then starting --demo ..."
  export MODEL_PATH TEXT_FILE OUT_WAV
  export MODEL_PATH="${MODEL_PATH}"
  run_cli_infer || true
  run_gradio_demo
elif [[ "${DO_DEMO}" -eq 1 ]]; then
  run_gradio_demo
elif [[ "${DO_INFER}" -eq 1 ]]; then
  export MODEL_PATH TEXT_FILE OUT_WAV
  export MODEL_PATH="${MODEL_PATH}"
  run_cli_infer
else
  cat <<EOF

${BOLD}Setup complete.${RESET}
Project dir: ${PROJECT_DIR}
Repo:        ${REPO_DIR}
Model path:  ${MODEL_PATH}
Venv:        ${VENV_DIR}
ffmpeg:      $(command -v ffmpeg || echo "${FFMPEG_DIR}/ffmpeg")

Next steps:
  - Launch Gradio demo:    ${BOLD}bash "$0" --demo${RESET}
  - Run CLI inference:     ${BOLD}bash "$0" --infer${RESET}
  - Choose a model:        ${BOLD}bash "$0" --model microsoft/VibeVoice-1.5B --demo${RESET}
  - Clean everything:      ${BOLD}bash "$0" --clean --force${RESET}

EOF
fi
