#!/bin/bash
# Oracle-vs-variant wall-clock comparison for the MWC manuscript.
#
# The main campaign ran timing at n=1600 with --oracle-max-m 1200, and the
# smallest instance there has m=1602, so t_oracle was null in all 144 rows and
# no wall-clock oracle speedup was measurable.  This task measures it at the
# sizes where the edge-removal oracle is actually affordable, on the SAME CPU
# (Xeon Gold 6248 / "rivanna") as every other published timing.
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH --constraint=rivanna
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -J mwc_oracle
#SBATCH -o /scratch/hs9hd/mwc_campaign_v2/logs/oracle_%A.out
#SBATCH -e /scratch/hs9hd/mwc_campaign_v2/logs/oracle_%A.err
set -euo pipefail

ROOT=/scratch/hs9hd/mwc_campaign_v2
CODE=$ROOT/repo/reference
OUT=$ROOT/results_oracle
mkdir -p "$OUT" "$ROOT/logs"
cd "$CODE"

echo "host=$(hostname)  job=${SLURM_JOB_ID}"
lscpu | grep -E "^Model name|^CPU\(s\):" | head -2
python3 -V
python3 -c "import networkx; print('networkx', networkx.__version__)"
echo "git: $(git -C "$CODE" rev-parse HEAD)"
echo "dirty: $(git -C "$CODE" status --porcelain | wc -l) files"
echo "----------------------------------------------------------------------"

# m<=2000 covers near_tree, sparse_er, grid and small_world at both n=400 and
# n=900; geometric (m=3216) and dense_er (m=11738) stay out of reach, which is
# itself the point -- the oracle is unaffordable there.
python3 campaign.py --out "$OUT" --repeats 5 timing \
        --sizes 400 900 --instances 24 --oracle-max-m 2000

echo "----------------------------------------------------------------------"
echo "oracle timing done"
