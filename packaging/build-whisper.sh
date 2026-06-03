#!/usr/bin/env bash
# Build the pinned whisper.cpp locally for development — same version and
# core flags the kdictate package vendors, so a `git clone` dev checkout
# runs the exact whisper that ships. No AUR, no llama.cpp-vulkan, no churn.
#
# After it finishes, export the printed path and backend.find_whisper_cpp()
# picks it up automatically:
#     export KDICTATE_WHISPER_CLI=<printed path>
set -euo pipefail

# Keep in sync with packaging/PKGBUILD _whisper_ver.
_whisper_ver="1.8.6"

root="$(cd "$(dirname "$0")/.." && pwd)"
out="${root}/.whisper"
src="${out}/whisper.cpp-${_whisper_ver}"

if [[ ! -d "$src" ]]; then
  mkdir -p "$out"
  curl -fsSL "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v${_whisper_ver}.tar.gz" \
    | tar -xz -C "$out"
fi

# A dev build runs only on this machine, so NATIVE=ON (tuned) plus a
# single self-contained static binary is simplest. The shipped package
# uses the portable multi-variant build instead (see packaging/PKGBUILD);
# transcription output is identical either way — only CPU dispatch differs.
cmake -B "${out}/build" -S "$src" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DWHISPER_USE_SYSTEM_GGML=OFF \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DWHISPER_SDL2=OFF \
  -DWHISPER_FFMPEG=OFF \
  -DWHISPER_BUILD_SERVER=OFF \
  -DWHISPER_BUILD_TESTS=OFF
cmake --build "${out}/build" --target whisper-cli

echo
echo "Built: ${out}/build/bin/whisper-cli"
echo "Dev use:  export KDICTATE_WHISPER_CLI=${out}/build/bin/whisper-cli"
