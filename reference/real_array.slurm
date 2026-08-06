#!/bin/bash
# Real-world networks: ONE array task per network.
#
# Cost varies by four orders of magnitude across these graphs
# (min{n,mu}*(m + n log n) runs from ~1e5 for lesmis to ~3.3e9 for
# osm-portland-drive), so running them in a single task would serialise the
# cheap ones behind the expensive ones and risk one walltime kill losing all
# ten results.  One task per network isolates the failures and lets the heavy
# road graphs use their own walltime.
#
# Timing protocol note: best-of-5 throughout; every network completes in
# well under a second, so there is no reason to economise.  The per-network JSON
# records `repeats` so the table can state the protocol.
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH --constraint=rivanna
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J mwc_real
#SBATCH -o /scratch/hs9hd/mwc_campaign_v2/logs/real_%A_%a.out
#SBATCH -e /scratch/hs9hd/mwc_campaign_v2/logs/real_%A_%a.err
set -euo pipefail

ROOT=/scratch/hs9hd/mwc_campaign_v2
CODE=$ROOT/code
OUT=$ROOT/results
mkdir -p "$OUT" "$ROOT/logs"
cd "$CODE"

# Ordered cheapest-first so early array slots return usable rows quickly.
NETS=(lesmis celegans-neural chicago-sketch-road usairport-2010 \
      openflights-air rome99-road uspowergrid-synth \
      chicago-regional-road sydney-road osm-portland-drive)
NET=${NETS[$SLURM_ARRAY_TASK_ID]}


# every network completes in under a second, so all get best-of-5
REPEATS=5

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID}  net=${NET}  repeats=${REPEATS}"
lscpu | grep -E "^Model name" | head -1
echo "git: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "----------------------------------------------------------------------"

python3 campaign.py --out "$OUT" --repeats "$REPEATS" real \
        --data "$ROOT/realnets" --max-n 60000 --only "$NET" --tag "$NET"

echo "task ${SLURM_ARRAY_TASK_ID} (${NET}) done"
