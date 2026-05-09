#!/usr/bin/env bash
# Convert + upload a queue of datasets. Conversions run sequentially (single
# GPU); uploads run in the background but are serialized via a flock so HF
# only sees one large-folder upload at a time.
#
# Usage:
#   _common/run_all.sh dataset_folder1 dataset_folder2 ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
CACHE_ROOT="$(cd "$REPO_ROOT"/.. && pwd)/lance_cache"
LOG_DIR="$CACHE_ROOT/.logs"
UPLOAD_LOCK="$CACHE_ROOT/.upload.lock"
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

PY="${PY:-/home/shadeform/.venv/bin/python}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <dataset_folder> [<dataset_folder>...]" >&2
    exit 1
fi

extract_repo_id() {
    "$PY" - <<EOF
import re, pathlib
src = pathlib.Path("$1/dataprep.py").read_text()
m = re.search(r'HF_REPO_ID\s*=\s*"([^"]+)"', src)
print(m.group(1) if m else "")
EOF
}

upload_serialized() {
    local repo_id="$1"
    local local_path="$2"
    (
        flock -x 9
        "$REPO_ROOT/_common/upload_and_cleanup.sh" "$repo_id" "$local_path"
    ) 9>"$UPLOAD_LOCK"
}
export -f upload_serialized

for folder in "$@"; do
    folder_abs="$REPO_ROOT/$folder"
    name="$(basename "$folder_abs")"
    repo_id="$(extract_repo_id "$folder_abs")"
    if [[ -z "$repo_id" ]]; then
        echo "[orchestrator] could not parse HF_REPO_ID from $folder_abs — skipping" >&2
        continue
    fi
    local_path="$CACHE_ROOT/${repo_id##*/}"

    echo "===== converting $name -> $repo_id ====="
    log="$LOG_DIR/$name.convert.log"
    if ! "$PY" "$folder_abs/dataprep.py" --overwrite >"$log" 2>&1; then
        echo "[orchestrator] convert failed for $name (see $log) — continuing with next dataset"
        continue
    fi

    if [[ -f "$folder_abs/HF_DATASET_CARD.md" && -d "$local_path" ]]; then
        cp "$folder_abs/HF_DATASET_CARD.md" "$local_path/README.md"
    fi

    if [[ ! -d "$local_path" ]]; then
        echo "[orchestrator] $local_path does not exist after conversion of $name — skipping upload"
        continue
    fi

    upload_log="$LOG_DIR/$name.upload.log"
    echo "[orchestrator] background upload -> $upload_log"
    REPO_ROOT="$REPO_ROOT" UPLOAD_LOCK="$UPLOAD_LOCK" \
        bash -c "upload_serialized '$repo_id' '$local_path'" >"$upload_log" 2>&1 &
    sleep 1
done

echo "[orchestrator] all conversions queued. tail $LOG_DIR/*.log for progress."
