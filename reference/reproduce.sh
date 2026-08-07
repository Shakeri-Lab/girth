#!/usr/bin/env bash
# Reproduce every claim in the paper from a FRESH CLONE of this repository.
#
#   git clone -b fix/exact-mwc-reference-implementation \
#       https://github.com/Shakeri-Lab/girth.git
#   cd girth/reference && ./reproduce.sh
#
# Nothing here reads a sibling checkout or a path outside this directory, with
# one declared exception: the Cholera figure needs the loop_modulus code and the
# case data, so that step is skipped unless LOOP_MODULUS points at it.
#
# Runtime: the correctness suite is seconds; the full campaign at the sizes used
# in the paper is roughly an hour on one core. Pass a smaller SIZES to smoke it.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
OUT=${OUT:-results}
TABLES=${TABLES:-tables}
SIZES=${SIZES:-1600}
INSTANCES=${INSTANCES:-24}
ABL_INSTANCES=${ABL_INSTANCES:-8}

echo "== environment"
$PY -V
$PY -c "import networkx; print('networkx', networkx.__version__)"
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: working tree is dirty; results will be stamped accordingly"
fi
mkdir -p "$OUT" "$OUT/oracle" "$TABLES"

echo
echo "== 1. correctness: 25 regressions + differential tests vs the oracle"
$PY -m pytest test_mwc.py -q

echo
echo "== 2. the proof witnesses (both must print gamma_0 = 1 and exit 0)"
$PY tightness_witness.py
$PY sharpness_witness.py

echo
echo "== 3. measurement campaign"
OUTDIR="$OUT" $PY tightness_witness.py >/dev/null   # emits tightness.json
$PY campaign.py --out "$OUT" --repeats 5 timing   --sizes $SIZES --instances "$INSTANCES" --oracle-max-m 1200
# The edge-removal oracle is unaffordable at $SIZES (the smallest instance there
# has m > 1200), so t_oracle is null in the run above.  The oracle comparison
# quoted in Section 5.7 is measured separately, at the sizes where the oracle
# actually terminates: m <= 2000 covers near_tree, sparse_er, grid and
# small_world at both n=400 and n=900.
$PY campaign.py --out "$OUT/oracle" --repeats 5 timing --sizes 400 900 --instances "$INSTANCES" --oracle-max-m 2000
$PY campaign.py --out "$OUT" --repeats 3 ablation --sizes $SIZES --instances "$ABL_INSTANCES" --a0-max-m 20000
$PY campaign.py --out "$OUT" --repeats 5 theta    --sizes 800 $SIZES --instances 3
$PY campaign.py --out "$OUT" --repeats 3 frontier --sizes 200 400 800 --instances 25
$PY campaign.py --out "$OUT" --repeats 1 argmin   --dims 4 5 6 7 8 9 10 12 14 16 18 20
for net in realnets/*.edges; do
  n=$(basename "$net" .edges)
  $PY campaign.py --out "$OUT" --repeats 5 real --data realnets --only "$n" --tag "$n"
done

echo
echo "== 4. the two measurement probes quoted in the text"
$PY probe_blocks.py
$PY probe_stats_bias.py

echo
echo "== 4b. the three hand-set tables (exactness, deletion rate, transversal)"
# These are typeset by hand in main.tex rather than generated, so this step
# prints the numbers to check them against; it does not write a .tex file.
$PY report.py
$PY transversal_study.py

echo
echo "== 5. tables and figures"
$PY make_tables.py --results "$OUT" --out "$TABLES"
$PY plot_argmin.py --json "$OUT/argmin.json" --out "$TABLES/argmin_counts.pdf"
if [ -n "${LOOP_MODULUS:-}" ]; then
  OUTDIR="$TABLES" $PY regen_cholera_figs.py
else
  echo "   (skipping the Cholera figures: set LOOP_MODULUS to the loop_modulus"
  echo "    checkout containing cholera_cases.gpkg to regenerate them)"
fi

echo
echo "== done.  Tables in $TABLES/, raw JSON in $OUT/."
echo "   Every table carries its source JSON, commit, job id and CPU as a LaTeX comment."
