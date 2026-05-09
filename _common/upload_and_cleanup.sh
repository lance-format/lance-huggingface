#!/usr/bin/env bash
# Upload a converted Lance dataset folder to the lance-format HF org via
# `hf upload-large-folder`, then delete the local copy on success.
#
# Usage:
#   _common/upload_and_cleanup.sh <repo_id> <local_path>
#
# Env:
#   HF_TOKEN — required if not already cached by `hf auth login`
#   KEEP_LOCAL=1 — skip the cleanup step (debugging)

set -euo pipefail

REPO_ID="${1:?repo_id required (e.g. lance-format/mnist-lance)}"
LOCAL_PATH="${2:?local path required}"

if [[ ! -d "$LOCAL_PATH" ]]; then
    echo "[upload] $LOCAL_PATH does not exist" >&2
    exit 1
fi

echo "[upload] $LOCAL_PATH -> https://huggingface.co/datasets/$REPO_ID"
START=$(date +%s)

# Make sure the repo exists (create=ok if missing). hf upload-large-folder
# will create it if needed too, but doing it explicitly gives a clearer error
# if the user lacks permissions.
hf repos create "$REPO_ID" --type dataset --exist-ok >/dev/null 2>&1 || true

hf upload-large-folder \
    "$REPO_ID" \
    "$LOCAL_PATH" \
    --repo-type dataset \
    --num-workers "${HF_UPLOAD_WORKERS:-8}" \
    --no-bars

ELAPSED=$(( $(date +%s) - START ))
echo "[upload] done in ${ELAPSED}s: https://huggingface.co/datasets/$REPO_ID"

if [[ "${KEEP_LOCAL:-0}" != "1" ]]; then
    echo "[upload] removing local copy: $LOCAL_PATH"
    rm -rf "$LOCAL_PATH"
    df -h "$(dirname "$LOCAL_PATH")" | tail -n 1
fi
