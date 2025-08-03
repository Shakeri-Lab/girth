# Performance Upgrade Roadmap

This document tracks the ongoing work to speed-up **girth** and then integrate those gains into **loop_modulus** while keeping both GitHub repositories clean and easy to collaborate on.

---
## 1  Groundwork

- [x] Standardise tooling (pytest, coverage, black/ruff, pre-commit)
- [x] Add GitHub Actions workflow
- [ ] Protect `master` with branch protection rules on GitHub (manual)
- [x] Freeze current public API; baseline tests recorded

## 2  Profiling & Baseline

| Graph | Size | Mean time (s) |
|-------|------|---------------|
| Grid  | 10×10 | 0.013 |
| Grid  | 20×20 | 0.044 |
| Grid  | 30×30 | 0.108 |
| Spatial | 50  | 0.027 |
| Spatial | 100 | 0.125 |
| Spatial | 150 | 0.312 |

Detailed cProfile outputs are in `profiling_results/`.

## 3  Optimisation Backlog

- [~] Refactor hot loops (reduce NetworkX look-ups, cache adjacency) – first pass completed (edge iteration optimized)
- [ ] Switch to faster heap/heapdict
- [ ] Optional Cython/C++ extension

---
_Maintain this file as the single source of truth for dev progress._
