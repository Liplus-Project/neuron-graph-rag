#!/usr/bin/env bash
set -euo pipefail

readonly PROTOCOL_ID="github-ngr-cross-encoder-precision-v4"
readonly PROTOCOL_COMMIT="a79e801483d656d401336198a5cc56887a286842"
readonly RUN_ROOT="/home/hal/ngr-experiments/github_cross_encoder_precision_v4"
readonly PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.11.15%2B20260807-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
readonly PYTHON_SIZE="30939767"
readonly PYTHON_SHA256="69dfac9d0f15a0b9281a38486f212cbf76421609228c184dc0d34a0533d57ba6"
readonly UV_URL="https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-unknown-linux-gnu.tar.gz"
readonly RUFF_URL="https://github.com/astral-sh/ruff/releases/download/0.16.4/ruff-x86_64-unknown-linux-gnu.tar.gz"

log_static_row() {
  local command_text="$1"
  local returncode="$2"
  local empty_sha
  local command_sha
  empty_sha="$(printf '' | sha256sum | cut -d' ' -f1)"
  command_sha="$(printf '%s' "$command_text" | sha256sum | cut -d' ' -f1)"
  sequence=$((sequence + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sequence" "$command_text" "$returncode" "$empty_sha" "$empty_sha" \
    "$command_sha" >> "$command_log"
}

run_logged() {
  local command_text
  local stdout_path
  local stderr_path
  local returncode
  local stdout_sha
  local stderr_sha
  local command_sha
  printf -v command_text '%q ' "$@"
  command_text="${command_text% }"
  sequence=$((sequence + 1))
  stdout_path="$RUN_ROOT/logs/command-${sequence}.stdout"
  stderr_path="$RUN_ROOT/logs/command-${sequence}.stderr"
  set +e
  "$@" >"$stdout_path" 2>"$stderr_path"
  returncode=$?
  set -e
  stdout_sha="$(sha256sum "$stdout_path" | cut -d' ' -f1)"
  stderr_sha="$(sha256sum "$stderr_path" | cut -d' ' -f1)"
  command_sha="$(printf '%s' "$command_text" | sha256sum | cut -d' ' -f1)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sequence" "$command_text" "$returncode" "$stdout_sha" "$stderr_sha" \
    "$command_sha" >> "$command_log"
  if [[ "$returncode" -ne 0 ]]; then
    tail -c 4000 "$stderr_path" >&2
    return "$returncode"
  fi
  cat "$stdout_path"
}

bootstrap_preflight() {
  local implementation_commit="$1"
  local windows_model_cache="$2"
  if [[ -e "$RUN_ROOT" ]]; then
    echo "run root already exists; v4 preflight is not retryable" >&2
    return 73
  fi
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT" || return $?
  command_log="$RUN_ROOT/bootstrap-commands.tsv"
  sequence=0
  : > "$command_log"
  mkdir "$RUN_ROOT/logs"
  log_static_row "test ! -e $RUN_ROOT" 0
  log_static_row "mkdir $RUN_ROOT" 0
  cat > "$RUN_ROOT/exclusive-create.json" <<EOF
{"absent_before_create":true,"exclusive_create_returncode":0,"protocol_id":"$PROTOCOL_ID","run_root":"$RUN_ROOT"}
EOF
  log_static_row "exclusive-create-marker $RUN_ROOT/exclusive-create.json" 0

  run_logged mkdir -p "$RUN_ROOT/downloads" "$RUN_ROOT/tools" "$RUN_ROOT/python"
  local shared_before
  shared_before="$(run_logged sha256sum /mnt/c/Users/smile/.ngrdb/knowledge.db)"
  printf '%s\n' "${shared_before%% *}" > "$RUN_ROOT/shared-db-before-preflight.sha256"
  log_static_row "shared-db-before-preflight $RUN_ROOT/shared-db-before-preflight.sha256" 0
  run_logged git clone https://github.com/Liplus-Project/neuron-graph-rag.git "$RUN_ROOT/source"
  run_logged git -C "$RUN_ROOT/source" checkout --detach "$implementation_commit"
  run_logged git -C "$RUN_ROOT/source" merge-base --is-ancestor "$PROTOCOL_COMMIT" HEAD
  run_logged curl --fail --location --output "$RUN_ROOT/downloads/cpython-3.11.15.tar.gz" "$PYTHON_URL"
  local actual_size
  local actual_sha_line
  actual_size="$(run_logged stat -c %s "$RUN_ROOT/downloads/cpython-3.11.15.tar.gz")"
  run_logged test "$actual_size" = "$PYTHON_SIZE"
  actual_sha_line="$(run_logged sha256sum "$RUN_ROOT/downloads/cpython-3.11.15.tar.gz")"
  run_logged test "${actual_sha_line%% *}" = "$PYTHON_SHA256"
  run_logged tar -xzf "$RUN_ROOT/downloads/cpython-3.11.15.tar.gz" -C "$RUN_ROOT/python" --strip-components=1
  run_logged curl --fail --location --output "$RUN_ROOT/downloads/uv-0.12.3.tar.gz" "$UV_URL"
  run_logged tar -xzf "$RUN_ROOT/downloads/uv-0.12.3.tar.gz" -C "$RUN_ROOT/tools" --strip-components=1
  run_logged "$RUN_ROOT/tools/uv" --version
  run_logged curl --fail --location --output "$RUN_ROOT/downloads/ruff-0.16.4.tar.gz" "$RUFF_URL"
  run_logged tar -xzf "$RUN_ROOT/downloads/ruff-0.16.4.tar.gz" -C "$RUN_ROOT/tools" --strip-components=1
  run_logged "$RUN_ROOT/tools/ruff" --version
  run_logged "$RUN_ROOT/python/bin/python3" --version
  run_logged "$RUN_ROOT/python/bin/python3" -m venv "$RUN_ROOT/.venv"
  run_logged "$RUN_ROOT/tools/uv" pip install --python "$RUN_ROOT/.venv/bin/python" --require-hashes -r "$RUN_ROOT/source/tests/fixtures/github_cross_encoder_precision_v4.requirements.lock"
  run_logged env PYTHONPATH="$RUN_ROOT/source/src" PYTHONUTF8=1 \
    "$RUN_ROOT/.venv/bin/python" -m neuron_graph_rag.cross_encoder_precision_v4_observation \
    model-copy-verify --source-cache "$windows_model_cache" \
    --cache "$RUN_ROOT/model-cache" --output "$RUN_ROOT/model-verification.json"
  run_logged env PYTHONPATH="$RUN_ROOT/source/src" PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$RUN_ROOT/model-cache" \
    HF_HUB_CACHE="$RUN_ROOT/model-cache" TORCH_HOME="$RUN_ROOT/torch-cache" \
    UV_CACHE_DIR="$RUN_ROOT/uv-cache" NO_PROXY='*' \
    "$RUN_ROOT/.venv/bin/python" -m neuron_graph_rag.cross_encoder_precision_v4_observation \
    preflight --external-root "$RUN_ROOT" --model-cache "$RUN_ROOT/model-cache"
}

run_observation() {
  if [[ ! -f "$RUN_ROOT/source/tests/evidence/github_cross_encoder_precision_v4/preflight.json" ]]; then
    echo "committed preflight evidence is unavailable in the ext4 source checkout" >&2
    return 66
  fi
  if [[ -e "$RUN_ROOT/source/runtime/github_cross_encoder_precision_v4/development.claim.json" \
     || -e "$RUN_ROOT/source/archive/github_cross_encoder_precision_v4/development.claim.json" ]]; then
    echo "development was already claimed; same-version retry is prohibited" >&2
    return 74
  fi
  env PYTHONPATH="$RUN_ROOT/source/src" PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$RUN_ROOT/model-cache" \
    HF_HUB_CACHE="$RUN_ROOT/model-cache" TORCH_HOME="$RUN_ROOT/torch-cache" \
    UV_CACHE_DIR="$RUN_ROOT/uv-cache" NO_PROXY='*' \
    "$RUN_ROOT/.venv/bin/python" -m neuron_graph_rag.cross_encoder_precision_v4_observation \
    run --external-root "$RUN_ROOT" --model-cache "$RUN_ROOT/model-cache"
}

case "${1:-}" in
  bootstrap-preflight)
    if [[ "$#" -ne 3 ]]; then
      echo "usage: $0 bootstrap-preflight IMPLEMENTATION_COMMIT WINDOWS_MODEL_CACHE" >&2
      exit 64
    fi
    bootstrap_preflight "$2" "$3"
    ;;
  run)
    if [[ "$#" -ne 1 ]]; then
      echo "usage: $0 run" >&2
      exit 64
    fi
    run_observation
    ;;
  *)
    echo "usage: $0 {bootstrap-preflight IMPLEMENTATION_COMMIT WINDOWS_MODEL_CACHE|run}" >&2
    exit 64
    ;;
esac
