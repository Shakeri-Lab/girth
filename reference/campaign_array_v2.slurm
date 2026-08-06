#!/bin/bash
# Experiments campaign for the MWC manuscript.  One array = one campaign.
#
# Hardware note: the published Table 5 was measured on one core of an Intel
# Xeon Gold 6248, which is the "rivanna" node feature (40-core).  We constrain
# to it so the rerun is comparable, and every task prints lscpu so the caption
# can be corrected if the constraint ever lands elsewhere.
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH --constraint=rivanna
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -J mwc_campaign
#SBATCH -o /scratch/hs9hd/mwc_campaign_v2/logs/task_%A_%a.out
#SBATCH -e /scratch/hs9hd/mwc_campaign_v2/logs/task_%A_%a.err
set -euo pipefail

ROOT=/scratch/hs9hd/mwc_campaign_v2
CODE=$ROOT/code
OUT=$ROOT/results
mkdir -p "$OUT" "$ROOT/logs"
cd "$CODE"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID}  job=${SLURM_ARRAY_JOB_ID}"
lscpu | grep -E "^Model name|^CPU\(s\):" | head -2
python3 -V
python3 -c "import networkx; print('networkx', networkx.__version__)"
echo "git: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "----------------------------------------------------------------------"

# One task id per experiment.  Each writes its own JSON into $OUT, so tasks
# never contend for a file and a single failure cannot corrupt the others.
case "${SLURM_ARRAY_TASK_ID}" in
  0)  # Table 5 rerun + the adaptive column, same protocol as bench_timing.py
      python3 campaign.py --out "$OUT" --repeats 5 timing \
              --sizes 400 900 1600 --oracle-max-m 1200 ;;
  1)  # A0-A7: what each acceleration actually buys
      python3 campaign.py --out "$OUT" --repeats 5 ablation \
              --sizes 400 900 1600 --a0-max-m 20000 ;;
  2)  # crossover sweep: fit theta from data rather than assume it
      python3 campaign.py --out "$OUT" --repeats 5 theta \
              --sizes 800 1600 --instances 3 \
              --ratios 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.5 ;;
  3)  # deferred: approximate-mode speed-accuracy frontier on multiscale
      python3 campaign.py --out "$OUT" --repeats 3 frontier \
              --sizes 200 400 800 --instances 25 ;;
  4)  # deferred: argmin counts for the withdrawn figure panel
      python3 campaign.py --out "$OUT" --repeats 1 argmin \
              --dims 4 5 6 7 8 9 10 12 14 16 18 20 ;;
  5)  # deferred: real networks beyond Cholera
      python3 campaign.py --out "$OUT" --repeats 3 real \
              --data "$ROOT/realnets" --max-n 60000 ;;
  *)  echo "no such task id"; exit 2 ;;
esac

echo "----------------------------------------------------------------------"
echo "task ${SLURM_ARRAY_TASK_ID} done"
