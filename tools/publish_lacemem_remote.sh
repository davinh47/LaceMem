#!/usr/bin/env bash
# Publish agentic_mem snapshots to https://github.com/davinh47/LaceMem.git
# with NO prior history: one commit on main, one on context-aware. Keeps upstream LICENSE.
#
# Usage (from anywhere):
#   bash /path/to/agentic_mem/tools/publish_lacemem_remote.sh
# Or:
#   SRC=/path/to/agentic_mem bash tools/publish_lacemem_remote.sh
#
# Requires: git, tar, network, push permission to the LaceMem repo.

set -euo pipefail

SRC="${SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
UPSTREAM="${UPSTREAM:-https://github.com/davinh47/LaceMem.git}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/lacemem-publish.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "[1/5] Fetch LICENSE from $UPSTREAM"
git clone --depth 1 "$UPSTREAM" "$WORK/upstream"
cp "$WORK/upstream/LICENSE" "$WORK/LICENSE"

echo "[2/5] Export main (tracked files only) -> $WORK/main"
mkdir "$WORK/main"
( cd "$SRC" && git archive main | tar -x -C "$WORK/main" )
cp "$WORK/LICENSE" "$WORK/main/LICENSE"
( cd "$WORK/main" && git init -b main && git remote add origin "$UPSTREAM" && git add -A )
( cd "$WORK/main" && git commit -m "Initial import: LaceMem-1.0 (main)" )

echo "[3/5] Export context-aware -> $WORK/ca"
mkdir "$WORK/ca"
( cd "$SRC" && git archive context-aware | tar -x -C "$WORK/ca" )
cp "$WORK/LICENSE" "$WORK/ca/LICENSE"
( cd "$WORK/ca" && git init -b context-aware && git remote add origin "$UPSTREAM" && git add -A )
( cd "$WORK/ca" && git commit -m "Initial import: LaceMem-1.0 (context-aware)" )

echo "[4/5] Force-push main (replaces remote history on default branch)"
( cd "$WORK/main" && git push -u origin main --force )

echo "[5/5] Force-push context-aware branch"
( cd "$WORK/ca" && git push -u origin context-aware --force )

echo "Done. Remote: $UPSTREAM (main + context-aware, single commit each)."
