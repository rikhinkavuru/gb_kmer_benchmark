#!/bin/bash
# Run a command politely: lowest scheduling priority (nice 20, yields to anything interactive) and at most
# $LOWCPU_THREADS (default 3) BLAS/OpenMP threads. Performance cores are allowed by default (user approved
# 2026-09-26: "optimize everything for quality and speed"). LOWCPU_BG=1 restores the stricter mode
# (background QoS = efficiency cores only, throttled I/O) for when the fanless MacBook Air runs hot.
# Usage: bibm/lowcpu.sh <command> [args...]
T="${LOWCPU_THREADS:-3}"
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
       VECLIB_MAXIMUM_THREADS="$T" NUMEXPR_NUM_THREADS="$T" LOWCPU_THREADS="$T"
if [ "${LOWCPU_BG:-0}" = "1" ]; then
  exec taskpolicy -b nice -n 20 "$@"
fi
exec nice -n 20 "$@"
