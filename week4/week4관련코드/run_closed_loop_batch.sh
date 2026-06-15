#!/usr/bin/env bash
# Closed-loop batch runner for the Week 4 assignment eval sets.
#
# Runs demo.py headless for every episode in a demo-json, saving one trajectory
# CSV per episode, then aggregates SR/OSR/SPL/nDTW/GoalDist with the metrics
# script. Scenes are read from an external SSD via --go2-physics-scene-root, so
# no main-disk space is needed for the kujiale USDs.
#
# Prereqs (do these first):
#   1. Plug in the external SSD and download the kujiale scenes for the episodes
#      you will run, laid out as  $SCENE_ROOT/<scan>/<scan>.usda  (+ Meshes/...).
#      (same layout as IAmGoodNavigator/kujiale_0010/)
#   2. Build the demo-json + index-map (already done for the 20-set):
#        python week4/build_closed_loop_demo_json.py --episode-list <sel.json> \
#            --split-json <fine_train.json.gz> [--split-json <fine_val.json.gz>] \
#            --out <demo.json>
#        python week4/make_index_map.py --demo-json <demo.json> --out <idx_map.json>
#
# Usage:
#   SCENE_ROOT=/mnt/ssd/vlnverse_scenes \
#   DEMO_JSON=/home/ad06/isaacsim/git/Deeplearning-final/data/closed_loop_eval/closed_loop_20_demo.json \
#   CKPT=/home/ad06/isaacsim/git/Deeplearning-final/outputs/week4/baseline/best_adapter \
#   N=20 WORKDIR=./myresults_cl20 EXTRA="" \
#   bash week4/run_closed_loop_batch.sh
#
# Run from inside /home/ad06/isaacsim/IAmGoodNavigator after sourcing the Isaac env.
set -u

: "${SCENE_ROOT:?set SCENE_ROOT to the external-SSD scenes dir}"
: "${DEMO_JSON:?set DEMO_JSON to the closed-loop demo json}"
: "${CKPT:?set CKPT to the LoRA adapter dir (or '' for zero-shot base)}"
: "${N:?set N to the number of episodes}"
WORKDIR="${WORKDIR:-./myresults_closed_loop}"
EXTRA="${EXTRA:-}"          # e.g. --go2-physics-week4-vln-trajectory-aware
MAXSTEPS="${MAXSTEPS:-120}"
# HEADLESS=1 (default) runs --headless; HEADLESS=0 runs windowed GUI on $DISPLAY.
# NOTE: the egocentric replicator camera returns empty frames under --headless,
# which makes the policy degenerate (one repeated action). Run with HEADLESS=0
# on a live X display (e.g. :1) so RTX renders the ego camera correctly.
HEADLESS="${HEADLESS:-1}"
if [ "$HEADLESS" = "1" ]; then HEADLESS_FLAG="--headless"; else HEADLESS_FLAG=""; export DISPLAY="${DISPLAY:-:1}"; fi

START="${START:-0}"        # first episode index to run (resume support)
mkdir -p "$WORKDIR"
echo "[batch] scenes=$SCENE_ROOT demo=$DEMO_JSON ckpt=$CKPT N=$N start=$START workdir=$WORKDIR headless=$HEADLESS display=${DISPLAY:-none}"

for i in $(seq "$START" $((N-1))); do
  echo "===== episode index $i / $((N-1)) ====="
  python demo.py --task fine --index "$i" $HEADLESS_FLAG \
    --work_dir "$WORKDIR" --agent go2 --go2-controller physics \
    --demo-json "$DEMO_JSON" \
    --go2-physics-scene-root "$SCENE_ROOT" \
    --go2-physics-week4-vln-checkpoint "$CKPT" \
    --go2-physics-week4-vln-max-steps "$MAXSTEPS" \
    $EXTRA \
    2>&1 | tee "$WORKDIR/ep_${i}.log"
done

echo "[batch] all episodes done. CSVs in $WORKDIR"
echo "[batch] aggregate with:"
echo "  python week4/eval_closed_loop_metrics.py --csv-dir $WORKDIR \\"
echo "    --demo-json $DEMO_JSON \\"
echo "    --index-map <idx_map.json> --out $WORKDIR/closed_loop_metrics.json"
