#!/usr/bin/env bash
# Build the pinned whisper.cpp locally for development — same version and
# core flags the kdictate package vendors, so a `git clone` dev checkout
# runs the exact whisper that ships. No AUR, no llama.cpp-vulkan, no churn.
#
# After it finishes, export KDICTATE_STT_WORKER using the printed path.
# The source installer also discovers .whisper/build/bin automatically.
set -euo pipefail

# Keep in sync with packaging/PKGBUILD _whisper_ver.
_whisper_ver="1.9.5"
_whisper_sha="ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720"

root="$(cd "$(dirname "$0")/.." && pwd)"
out="${root}/.whisper"
src="${out}/whisper.cpp-${_whisper_ver}"

if [[ ! -d "$src" ]]; then
  mkdir -p "$out"
  archive="${out}/whisper.cpp-${_whisper_ver}.tar.gz"
  curl -fsSL "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v${_whisper_ver}.tar.gz" \
    -o "$archive"
  printf '%s  %s\n' "$_whisper_sha" "$archive" | sha256sum --check -
  tar -xzf "$archive" -C "$out"
fi

# A dev build tunes CPU kernels for this machine. Shared libraries stay
# beside the worker; the package uses portable CPU variants instead.
cmake -B "${out}/build" -S "$src" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DWHISPER_USE_SYSTEM_GGML=OFF \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DWHISPER_SDL2=OFF \
  -DWHISPER_FFMPEG=OFF \
  -DWHISPER_BUILD_SERVER=OFF \
  -DWHISPER_BUILD_TESTS=OFF
cmake --build "${out}/build" --target whisper-cli parakeet-cli
bash "${root}/packaging/build-stt-worker.sh" "$src" "${out}/build"

echo
echo "Built: ${out}/build/bin/whisper-cli"
echo "Dev use:  export KDICTATE_WHISPER_CLI=${out}/build/bin/whisper-cli"
echo "Resident: export KDICTATE_STT_WORKER=${out}/build/bin/kdictate-stt-worker"
