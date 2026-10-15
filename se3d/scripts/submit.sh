#!/usr/bin/env bash
# usage: se3d/scripts/submit.sh NAME HOURS MEM_GB "command ..." [min_vram_gb]
# Uses ONLY scavenger / vulcan-scavenger, prefers rtxa5000 / rtxa4000 (fast to
# place). Picks the allowed (partition, gpu) pair with the most idle nodes.
# min_vram_gb > 24 falls back to rtxa6000.
set -eo pipefail
NAME=$1; HOURS=$2; MEM=$3; CMD=$4; MINVRAM=${5:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /fs/nexus-scratch/vinayakg/se3d/logs/slurm

if [ "$MINVRAM" -le 16 ]; then TYPES="rtxa5000 rtxa4000"
elif [ "$MINVRAM" -le 24 ]; then TYPES="rtxa5000"
else TYPES="rtxa6000"; fi

best=""; bestn=-1
for P in scavenger vulcan-scavenger; do
  for T in $TYPES; do
    n=$(sinfo -p $P -h -t idle,mix -o "%G %D" 2>/dev/null | grep -c "gpu:$T" || true)
    i=$(sinfo -p $P -h -t idle -o "%G %D" 2>/dev/null | { grep "gpu:$T" || true; } | awk '{s+=$2} END{print s+0}' || true)
    score=$(( i * 10 + n ))
    if [ "$score" -gt "$bestn" ]; then bestn=$score; best="$P $T"; fi
  done
done
read P T <<< "$best"
# if my jobs on scavenger are blocked by the per-user GPU limit, use vulcan-scavenger
if [ "$P" = "scavenger" ] && squeue -u $USER -h -p scavenger -t PD -o "%R" | grep -q QOSMaxGRESPerUser; then
  P=vulcan-scavenger
  for T2 in $TYPES; do
    if sinfo -p vulcan-scavenger -h -t idle,mix -o "%G" | grep -q "gpu:$T2"; then T=$T2; break; fi
  done
fi
if [ "$P" = "scavenger" ]; then ACC=scavenger; QOS=scavenger; else ACC=vulcan; QOS=vulcan-scavenger; fi

jid=$(SE3D_CMD="$CMD" sbatch --parsable --job-name="$NAME" --partition=$P --account=$ACC --qos=$QOS \
  --gres=gpu:$T:1 --cpus-per-task=4 --mem=${MEM}G --time=$(printf '%02d:00:00' $HOURS) \
  --export=ALL "$HERE/job.sbatch")
echo "$jid $P $T $NAME"
