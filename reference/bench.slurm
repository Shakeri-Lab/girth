#!/bin/bash
#SBATCH -A shakeri-lab
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 01:30:00
#SBATCH -J mwc_bench
#SBATCH -o /project/shakeri-lab/graph_alg/girth/reference/bench_%j.out
set -e
cd /project/shakeri-lab/graph_alg/girth/reference
echo "host=$(hostname)  python=$(python3 -V 2>&1)"
python3 -c "import networkx;print('networkx', networkx.__version__)"
lscpu | grep -E "^Model name|^CPU\(s\):" | head -2
echo "-----"
SIZES=${SIZES:-400,900,1600} REPEATS=5 ORACLE_MAX_M=1200 python3 bench_timing.py
