#!/bin/sh
set -eu

VISIONPROOF_WORK="${1:-$HOME/.codex/work/visionproof}"
VISIONPROOF_PROVIDER="${VISIONPROOF_PROVIDER:-rules}"
VISIONPROOF_MODEL="${VISIONPROOF_MODEL:-}"
mkdir -p "$VISIONPROOF_WORK"
uv run vision-proof prepare --data "$VISIONPROOF_WORK/data" --limit 80
if [ ! -f "$VISIONPROOF_WORK/data/manifest-fresh.json" ]; then
    uv run vision-proof prepare-fresh --original "$VISIONPROOF_WORK/data/manifest.json" \
        --out "$VISIONPROOF_WORK/data/manifest-fresh.json" --limit 64
fi
VISIONPROOF_RUN="$VISIONPROOF_WORK/run-$(date -u +%Y%m%dT%H%M%S)-$$"
if [ "$VISIONPROOF_PROVIDER" = bedrock ]; then
    test -n "$VISIONPROOF_MODEL" || { echo 'Set VISIONPROOF_MODEL to an accessible Bedrock model.' >&2; exit 1; }
    uv run --extra cloud vision-proof proof --manifest "$VISIONPROOF_WORK/data/manifest-fresh.json" \
        --out "$VISIONPROOF_RUN" --provider bedrock --model "$VISIONPROOF_MODEL" --budget 4
    exec uv run --extra cloud vision-proof serve --manifest "$VISIONPROOF_WORK/data/manifest-fresh.json" \
        --run "$VISIONPROOF_RUN" --evidence-root "$VISIONPROOF_WORK" --model "$VISIONPROOF_MODEL"
fi
uv run vision-proof proof --manifest "$VISIONPROOF_WORK/data/manifest-fresh.json" \
    --out "$VISIONPROOF_RUN" --provider rules --budget 4
exec uv run vision-proof serve --manifest "$VISIONPROOF_WORK/data/manifest-fresh.json" \
    --run "$VISIONPROOF_RUN" --evidence-root "$VISIONPROOF_WORK"
