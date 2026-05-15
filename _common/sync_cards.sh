#!/usr/bin/env bash
# Push every HF_DATASET_CARD.md in this repo to its corresponding dataset repo
# on the Hugging Face Hub (uploaded as README.md).
#
# Usage:
#   _common/sync_cards.sh           # upload all
#   _common/sync_cards.sh cifar10   # upload only the given dirs (one or more)
#
# Env:
#   HF_TOKEN  — required if not already cached by `hf auth login`
#   DRY_RUN=1 — print what would be uploaded without calling `hf upload`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# directory (relative to repo root) -> HF dataset repo id
declare -a CARDS=(
    "cifar10                       lance-format/cifar10-lance"
    "mnist                         lance-format/mnist-lance"
    "fashion_mnist                 lance-format/fashion-mnist-lance"
    "food101                       lance-format/food101-lance"
    "oxford_pets                   lance-format/oxford-pets-lance"
    "stanford_cars                 lance-format/stanford-cars-lance"
    "eurosat                       lance-format/eurosat-lance"
    "imagenet1k_val                lance-format/imagenet-1k-val-lance"
    "ade20k                        lance-format/ade20k-lance"
    "pascal_voc_2012               lance-format/pascal-voc-2012-segmentation-lance"
    "kitti                         lance-format/kitti-2d-detection-lance"
    "coco_captions_2017            lance-format/coco-captions-2017-lance"
    "coco_detection_2017           lance-format/coco-detection-2017-lance"
    "flickr30k                     lance-format/flickr30k-lance"
    "laion-1M                      lance-format/laion-1m"
    "chartqa                       lance-format/chartqa-lance"
    "docvqa                        lance-format/docvqa-lance"
    "textvqa                       lance-format/textvqa-lance"
    "vqav2                         lance-format/vqav2-lance"
    "gqa                           lance-format/gqa-testdev-balanced-lance"
    "squad_v2                      lance-format/squad-v2-lance"
    "triviaqa                      lance-format/trivia-qa-lance"
    "hotpotqa                      lance-format/hotpotqa-distractor-lance"
    "natural_questions             lance-format/natural-questions-val-lance"
    "ms_marco                      lance-format/ms-marco-v2.1-lance"
    "librispeech                   lance-format/librispeech-clean-lance"
    "fineweb_edu                   lance-format/fineweb-edu"
    "openvid_hf                    lance-format/openvid-lance"
    "lerobot/pusht                 lance-format/lerobot-pusht-lance"
    "lerobot/xvla-soft-fold        lance-format/lerobot-xvla-soft-fold"
)

# Optional filter: only sync the dirs passed on the command line.
FILTER=("$@")
should_sync() {
    local dir="$1"
    if [[ ${#FILTER[@]} -eq 0 ]]; then
        return 0
    fi
    for f in "${FILTER[@]}"; do
        [[ "$dir" == "$f" ]] && return 0
    done
    return 1
}

failures=()
for entry in "${CARDS[@]}"; do
    read -r DIR REPO <<<"$entry"
    CARD="$REPO_ROOT/$DIR/HF_DATASET_CARD.md"

    should_sync "$DIR" || continue

    if [[ ! -f "$CARD" ]]; then
        echo "[skip] $DIR (no HF_DATASET_CARD.md)" >&2
        continue
    fi

    echo "[sync] $DIR -> https://huggingface.co/datasets/$REPO"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    if ! hf upload "$REPO" "$CARD" README.md \
            --repo-type dataset \
            --commit-message "Update README with LanceDB examples"; then
        echo "[fail] $DIR" >&2
        failures+=("$DIR")
    fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
    echo
    echo "Failed: ${failures[*]}" >&2
    exit 1
fi
