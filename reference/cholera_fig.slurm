#!/bin/bash
# Regenerate the Cholera rho* figure from the certified solver, as vector PDF.
# ~20s of real work (632 QP solves), so it is a batch job, not a login-node run.
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J cholera_fig
#SBATCH -o /scratch/hs9hd/cholera_fig/regen_%j.out
#SBATCH -e /scratch/hs9hd/cholera_fig/regen_%j.err
set -euo pipefail

OUTDIR=/scratch/hs9hd/cholera_fig
mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "host=$(hostname)"
python3 -V
echo "----------------------------------------------------------------------"
OUTDIR="$OUTDIR" python3 "$OUTDIR/regen_cholera_rho.py"
echo "----------------------------------------------------------------------"
ls -la "$OUTDIR"/cholera_rho_certified.*
