#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH --constraint=rivanna
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J mwc_bench_accel
#SBATCH -o /scratch/hs9hd/mwc_certified_pruning/logs/accel_bench_%j.out
#SBATCH -e /scratch/hs9hd/mwc_certified_pruning/logs/accel_bench_%j.err
# ==============================================================================
# Milestone 2 Acceleration Verification & Benchmarking SLURM Job
# Hardware: Intel Xeon Gold 6248 (Standardized on Rivanna)
# Compares: Pure Python baseline vs. C++/Cython compiled kernel
# ==============================================================================

set -euo pipefail

ROOT=/scratch/hs9hd/mwc_certified_pruning
VENV=$ROOT/venv
OUT=$ROOT/results/acceleration
LOGS=$ROOT/logs

mkdir -p "$OUT" "$LOGS"
source "$VENV/bin/activate"

# Optional GCC environment if dynamic recompilation needed
module load gcc/11.4.0 2>/dev/null || true

export PYTHONPATH="$ROOT/girth:$ROOT:${PYTHONPATH:-}"

echo "======================================================================"
echo "Starting Milestone 2 Acceleration Benchmark on Rivanna"
echo "Host: $(hostname) | Job ID: ${SLURM_JOB_ID:-local}"
lscpu | grep -E "^Model name" | head -1
python3 -V
python3 -c "import networkx, numpy; print(f'NetworkX {networkx.__version__}, NumPy {numpy.__version__}')"
python3 -c "
try:
    import girth.c_extensions.fast_mwc
    print('Cython fast_mwc extension: AVAILABLE (COMPILED)')
except ImportError as e:
    print(f'Cython fast_mwc extension: FALLBACK (Not compiled or import failed: {e})')
"
echo "======================================================================"

python3 "$ROOT/girth/benchmark_acceleration.py" \
    --families all \
    --trials 5 \
    --warmup 1 \
    --dense-sizes 20 40 60 80 100 150 200 300 500 \
    --sparse-sizes 50 100 200 500 1000 2000 5000 \
    --realnets all \
    --datasets-dir "$ROOT/datasets/realnets" \
    --weight-type fractional \
    --tolerance 1e-9 \
    --out-dir "$OUT"

echo "======================================================================"
echo "Acceleration Benchmark Finished Successfully"
echo "Artifacts saved in: $OUT"
echo "======================================================================"
