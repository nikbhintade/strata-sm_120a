#!/usr/bin/env bash
# Automated parity + performance regression gate for strata-sm_120a on RTX 5060 Ti 16 GB.
set -euo pipefail

ROOT="/home/nikhil/Documents/ai-content/inkling-2"
BUILD="$ROOT/Strata/engine/build-inkling"
BIN_MAIN="$BUILD/src/inkling2/strata-inkling"
BIN_CUDA="$BUILD/src/inkling2/cuda/strata-inkling-cuda"
MODEL="$ROOT/models/Inkling-2-Mini-SQ-15.strata"
REF="$ROOT/bench_refs/ref_sq15_32.bin"

echo "=== 1. PARITY GATE (--kld-eval against baseline ref_sq15_32.bin) ==="
"$BIN_MAIN" --kld-eval "$REF" --native "$MODEL" --kv fp8 --kv-offload on --draft off

echo "=== 2. KERNEL PROFILE (--prof, 100% VRAM, prompt=512 & 2048, chunk=512, gen=64) ==="
"$BIN_CUDA" --model "$MODEL" --prompt-len 512 --chunk 512 --gen 64 --ctx 8192 --kv-fp8 --prof --draft off
"$BIN_CUDA" --model "$MODEL" --prompt-len 2048 --chunk 512 --gen 16 --ctx 8192 --kv-fp8 --prof --draft off

echo "=== 3. STRATA-INKLING-CUDA FULL SPEED + DSPARK + PARITY (prompt=512, chunk=512, gen=128) ==="
"$BIN_CUDA" --model "$MODEL" --prompt-len 512 --chunk 512 --gen 128 --ctx 8192 --kv-fp8 --parity

echo "=== 4. STRATA-INKLING --bench (512 / 2048 / 4096, DSpark ON & OFF, kv=fp8) ==="
for plen in 512 2048 4096; do
  echo "--- prompt=$plen draft=dspark ---"
  "$BIN_MAIN" --bench --native "$MODEL" --kv fp8 --chunk 512 --prompt-len "$plen" --gen 128 --ctx 8192 --draft dspark
  echo "--- prompt=$plen draft=off ---"
  "$BIN_MAIN" --bench --native "$MODEL" --kv fp8 --chunk 512 --prompt-len "$plen" --gen 128 --ctx 8192 --draft off
done
