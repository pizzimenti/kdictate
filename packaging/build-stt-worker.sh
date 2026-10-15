#!/usr/bin/env bash
# Compile against the same pinned libraries as the CLI. No separate GGML ABI.
set -euo pipefail
src="${1:?whisper source directory}"
bld="${2:?whisper build directory}"
out="${3:-${bld}/bin/kdictate-stt-worker}"
root="$(cd "$(dirname "$0")/.." && pwd)"
"${CXX:-c++}" -std=c++17 -O2 \
    -I"${src}/include" -I"${src}/ggml/include" \
    "${root}/native/stt-worker.cpp" \
    -L"${bld}/bin" -L"${bld}/src" -L"${bld}/ggml/src" \
    -lwhisper -lparakeet -lggml -lggml-base \
    '-Wl,-rpath,$ORIGIN:$ORIGIN/../src:$ORIGIN/../ggml/src:$ORIGIN/../lib' \
    -o "$out"
