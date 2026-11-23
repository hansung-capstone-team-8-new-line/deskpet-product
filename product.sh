#!/bin/sh
# Jetson Orin Nano production entry point (dash-compatible /bin/sh).
# - Clones missing repositories (deskpet-pipeline, deskpet-hardware, deskpet-harness) as relative siblings.
# - Bootstraps Swift toolchain/runtime if `swift` is not installed.
# - Configures Models/native.jetson.local.json without Python:
#     Default (Option 1): llama.cpp (router, output) + Zipformer-Korean 72M INT8 (stt) + Qwen3-TTS 0.6B INT4 (tts).
#     --low-memory (Option 2): llama.cpp (router, output) + Zipformer-Korean 174M INT8 (stt) + MeloTTS-Korean INT8 (tts).
# - Builds deskpet-pipeline/native (`libdeskpet_inference.so`) with llama.cpp (CUDA),
#   whisper.cpp, sherpa-onnx, and ONNX Runtime TTS (`tokenizers-cpp`).
# - Supports `--use-audio-gpu` to route STT and TTS through GPUExecutor (`DESKPET_USE_AUDIO_GPU=1`) and configure `stt`/`tts` device as `cuda`.
set -eu

script_dir=$(cd -P "$(dirname "$0")" && pwd)
workspace_dir=$(cd -P "$script_dir/.." && pwd)

use_audio_gpu=0
low_memory=0
has_real=0
has_headless=0
has_no_build=0
has_help=0

nargs=$#
i=0
while [ "$i" -lt "$nargs" ]; do
  arg=$1
  shift
  case "$arg" in
    --use-audio-gpu)
      use_audio_gpu=1
      ;;
    --low-memory)
      low_memory=1
      ;;
    --real-hardware)
      has_real=1
      set -- "$@" "$arg"
      ;;
    --headless)
      has_headless=1
      set -- "$@" "$arg"
      ;;
    --no-build)
      has_no_build=1
      set -- "$@" "$arg"
      ;;
    --help)
      has_help=1
      set -- "$@" "$arg"
      ;;
    *)
      set -- "$@" "$arg"
      ;;
  esac
  i=$((i + 1))
done

if [ "$has_help" = "1" ]; then
  echo "Usage: $0 [--low-memory] [--use-audio-gpu] [deskpet-demo --jetson options...]"
  echo "  --low-memory      Use Option 2: Zipformer-Korean 174M INT8 STT + MeloTTS-Korean INT8 TTS (~3.9-4.3 GB total)"
  echo "  (default)         Use Option 1: Zipformer-Korean 72M INT8 STT + Qwen3-TTS 0.6B INT4 TTS (~4.5-4.9 GB total)"
  echo "  --use-audio-gpu   Run STT and TTS via GPUExecutor (DESKPET_USE_AUDIO_GPU=1) with STT/TTS device=cuda"
fi

if [ "$low_memory" = "1" ]; then
  export DESKPET_LOW_MEMORY=1
  echo "[product.sh] Low-memory mode enabled (Option 2: Zipformer-Korean 174M INT8 + MeloTTS-Korean INT8)"
fi

if [ "$use_audio_gpu" = "1" ]; then
  export DESKPET_USE_AUDIO_GPU=1
  echo "[product.sh] Audio GPU execution enabled (DESKPET_USE_AUDIO_GPU=1)"
fi

REPO_BASE="https://github.com/hansung-capstone-team-8-new-line"
for repo in deskpet-pipeline deskpet-hardware deskpet-harness; do
  if [ ! -d "$workspace_dir/$repo" ]; then
    echo "[product.sh] Cloning $REPO_BASE/$repo.git -> $workspace_dir/$repo"
    git clone "$REPO_BASE/$repo.git" "$workspace_dir/$repo"
  fi
done

cd -P "$workspace_dir/deskpet-pipeline"
pipeline_root=$PWD

SWIFT_ROOT="$workspace_dir/.swift-toolchain"
export PATH="$SWIFT_ROOT/usr/bin:$HOME/.local/swift/usr/bin:$HOME/.cargo/bin:$PATH:/usr/local/cuda/bin:/usr/local/bin:$HOME/.local/share/pnpm:$HOME/Library/pnpm"

ensure_swift_runtime() {
  if command -v swift >/dev/null 2>&1; then
    return 0
  fi

  echo "[product.sh] Swift runtime not found. Preparing Swift toolchain in $SWIFT_ROOT ..."
  SUDO=""
  if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  fi

  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -y
    $SUDO apt-get install -y \
      binutils curl git gnupg2 libc6-dev libcurl4-openssl-dev libedit2 \
      libgcc-s1 libpython3-dev libsqlite3-0 libstdc++6 libxml2-dev \
      libz3-dev pkg-config tzdata unzip zlib1g-dev build-essential cmake ninja-build
  fi

  arch=$(uname -m)
  case "$arch" in
    aarch64|arm64) swift_arch="aarch64" ;;
    x86_64|amd64)  swift_arch="x86_64" ;;
    *) echo "[product.sh] Unsupported architecture for Swift: $arch" >&2; exit 1 ;;
  esac

  ubuntu_ver="22.04"
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ] && [ -n "${VERSION_ID:-}" ]; then
      ubuntu_ver="$VERSION_ID"
    fi
  fi
  ubuntu_tag=$(printf '%s' "$ubuntu_ver" | tr -d '.')

  swift_ver="6.4.2"
  if [ "$swift_arch" = "aarch64" ]; then
    tar_name="swift-${swift_ver}-RELEASE-ubuntu${ubuntu_ver}-aarch64"
    url="https://download.swift.org/swift-${swift_ver}-release/ubuntu${ubuntu_tag}-aarch64/swift-${swift_ver}-RELEASE/${tar_name}.tar.gz"
  else
    tar_name="swift-${swift_ver}-RELEASE-ubuntu${ubuntu_ver}"
    url="https://download.swift.org/swift-${swift_ver}-release/ubuntu${ubuntu_tag}/swift-${swift_ver}-RELEASE/${tar_name}.tar.gz"
  fi

  mkdir -p "$SWIFT_ROOT"
  tmp_tar="$SWIFT_ROOT/swift.tar.gz"
  if curl -fL "$url" -o "$tmp_tar"; then
    tar -xzf "$tmp_tar" --strip-components=1 -C "$SWIFT_ROOT"
    rm -f "$tmp_tar"
  else
    echo "[product.sh] Prebuilt tarball unavailable for Ubuntu $ubuntu_ver ($swift_arch); installing via swiftly..."
    rm -f "$tmp_tar"
    curl -L "https://swiftlang.github.io/swiftly/swiftly-${arch}-unknown-linux-gnu" -o "$SWIFT_ROOT/swiftly"
    chmod +x "$SWIFT_ROOT/swiftly"
    "$SWIFT_ROOT/swiftly" init --quiet-shell-followup -y "${swift_ver}"
    if [ -d "$HOME/.local/share/swiftly/toolchains" ]; then
      export PATH="$HOME/.local/share/swiftly/bin:$PATH"
    fi
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "[product.sh] Failed to prepare Swift runtime." >&2
    exit 1
  fi
}

ensure_rust_runtime() {
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    return 0
  fi
  echo "[product.sh] Rust toolchain not found. Installing minimal rustup for official tokenizers-cpp ..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[product.sh] Failed to prepare Rust cargo runtime." >&2
    exit 1
  fi
}

ensure_swift_runtime

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SWIFT_ROOT/usr/lib/swift/linux:$PWD/.build/native:$PWD/.build/native/llama/src:$PWD/.build/native/llama/ggml/src:$PWD/.build/native/llama/ggml/src/ggml-cuda:$PWD/.build/native/whisper/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

export DESKPET_HARNESS_DIR="../deskpet-harness"
export DESKPET_HARDWARE_DIR="../deskpet-hardware"

fetch_file() {
  url=$1
  dest=$2
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[product.sh] Downloading $(basename "$dest") ..."
  curl --fail --location --retry 3 "$url" --output "$dest.part"
  mv "$dest.part" "$dest"
}

ensure_models() {
  if [ "$has_help" = "1" ]; then
    return 0
  fi
  mkdir -p "Models"

  # 1) Directly downloadable shared runtime models.
  fetch_file "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_0.gguf" "Models/output/qwen-2b-q4_0.gguf"
  fetch_file "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" "Models/vad/silero_vad.onnx"
  fetch_file "https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc/resolve/d1c0233fc013ae8bd879d7b61f2bec69aa3af4c9/tokenizer.json" "Models/emotion/tokenizer.json"

  # 2) Speech models (Option 1 default vs Option 2 --low-memory).
  if [ "$low_memory" = "1" ]; then
    stt_dir="Models/stt/icefall-asr-ko-streaming-zipformer-174m"
    if [ ! -f "$stt_dir/tokens.txt" ]; then
      base_url="https://huggingface.co/kangkyu/icefall-asr-ko-streaming-zipformer-174m/resolve/main"
      fetch_file "$base_url/tokens.txt" "$stt_dir/tokens.txt"
      fetch_file "$base_url/encoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/encoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
      fetch_file "$base_url/decoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/decoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
      fetch_file "$base_url/joiner-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/joiner-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
    fi
    tts_dir="Models/tts/melotts-kr-onnx"
    if [ ! -f "$tts_dir/melotts_kr_int8.onnx" ] || [ ! -f "$tts_dir/symbols.json" ]; then
      base_url="https://huggingface.co/gnyong/melotts-kr-onnx/resolve/main"
      fetch_file "$base_url/symbols.json" "$tts_dir/symbols.json"
      fetch_file "$base_url/melotts_kr_int8.onnx" "$tts_dir/melotts_kr_int8.onnx"
    fi
  else
    stt_dir="Models/stt/sherpa-onnx-zipformer-korean-2024-06-24"
    if [ ! -f "$stt_dir/tokens.txt" ]; then
      archive="$pipeline_root/.build/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2"
      mkdir -p "$pipeline_root/.build" "Models/stt"
      fetch_file "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2" "$archive"
      tar -xjf "$archive" -C "Models/stt"
      rm -f "$archive"
    fi
    tts_dir="Models/tts/qwen3-tts-12hz-0.6b-customvoice"
    if [ ! -f "$tts_dir/config.json" ] && command -v unzip >/dev/null 2>&1; then
      for zip_cand in "$pipeline_root/Qwen3-Models.zip" "$script_dir/Qwen3-Models.zip"; do
        if [ -f "$zip_cand" ]; then
          echo "[product.sh] Extracting $tts_dir from $zip_cand ..."
          unzip -oq "$zip_cand" "Models/tts/qwen3-tts-12hz-0.6b-customvoice/*" -d "."
          break
        fi
      done
    fi
  fi

  for req_local in \
    "Models/router/backbone-merged-q4_0.gguf" \
    "Models/router/heads-f16.gguf" \
    "Models/emotion/model.onnx"; do
    if [ ! -f "$req_local" ]; then
      echo "[product.sh] Note: locally-built model missing (not directly downloadable): $req_local" >&2
    fi
  done
  if [ "$low_memory" != "1" ] && [ ! -f "Models/tts/qwen3-tts-12hz-0.6b-customvoice/config.json" ]; then
    echo "[product.sh] Note: Option 1 TTS bundle missing: Models/tts/qwen3-tts-12hz-0.6b-customvoice (place Qwen3-Models.zip or directory, or run with --low-memory)." >&2
  fi
}

# 1. Download ready-to-use missing models, prepare Option 1 / Option 2 speech models, and configure Models/native.jetson.local.json via Swift.
ensure_models

if [ ! -f "Models/native.jetson.local.json" ] && [ -f "Models/native.jetson.example.json" ]; then
  cp "Models/native.jetson.example.json" "Models/native.jetson.local.json"
fi

if [ -f "Models/native.jetson.local.json" ]; then
  cfg_args=""
  if [ "$use_audio_gpu" = "1" ]; then cfg_args="$cfg_args --use-audio-gpu"; fi
  if [ "$low_memory" = "1" ]; then cfg_args="$cfg_args --low-memory"; fi
  swift "$script_dir/configure_jetson_json.swift" $cfg_args
fi

download_ort_headers_from_nuget() {
  sherpa_dir=$1
  ort_include="$sherpa_dir/include/onnxruntime/core/session"
  mkdir -p "$ort_include"
  ort_ver="1.28.0"
  ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_error_code.h onnxruntime_ep_c_api.h"
  archive_name=""
  if [ -f "$sherpa_dir/.archive" ]; then
    archive_name=$(cat "$sherpa_dir/.archive")
  fi
  case "$archive_name" in
    *onnxruntime-1.18.1*)
      ort_ver="1.18.1"
      ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h"
      ;;
    *onnxruntime-1.18.0*)
      ort_ver="1.18.0"
      ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h"
      ;;
  esac
  nupkg="$pipeline_root/.build/ort-$ort_ver.nupkg"
  curl --fail --location --retry 3 \
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ort_ver/microsoft.ml.onnxruntime.$ort_ver.nupkg" \
    --output "$nupkg.part"
  mv "$nupkg.part" "$nupkg"
  if command -v unzip >/dev/null 2>&1; then
    for header in $ort_headers; do
      unzip -oqj "$nupkg" "build/native/include/$header" -d "$ort_include"
    done
  else
    echo "[product.sh] unzip is required to extract onnxruntime headers from $nupkg" >&2
    exit 1
  fi
  rm -f "$nupkg"
}

ensure_native_deps() {
  # TTS runs on CUDA GPU by default, so always fetch the GPU build of sherpa-onnx + onnxruntime.
  sherpa_mode="gpu"
  cur_sherpa_mode=""
  if [ -f "$pipeline_root/.build/native-deps/sherpa/.mode" ]; then
    cur_sherpa_mode=$(cat "$pipeline_root/.build/native-deps/sherpa/.mode")
  fi
  if [ ! -f "$pipeline_root/.build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so" ] || \
     [ ! -f "$pipeline_root/.build/native-deps/sherpa/include/onnxruntime/core/session/onnxruntime_cxx_api.h" ] || \
     [ "$cur_sherpa_mode" != "$sherpa_mode" ]; then
    if ! DESKPET_USE_AUDIO_GPU=1 sh "$pipeline_root/tools/setup-native-deps-linux.sh"; then
      if [ -f "$pipeline_root/.build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so" ] && \
         [ ! -f "$pipeline_root/.build/native-deps/sherpa/include/onnxruntime/core/session/onnxruntime_cxx_api.h" ]; then
        echo "[product.sh] Falling back to NuGet for onnxruntime headers ..."
        download_ort_headers_from_nuget "$pipeline_root/.build/native-deps/sherpa"
      else
        exit 1
      fi
    fi
  fi
}

# 2. Prepare Swift/CMake workspace metadata.
sh tools/prepare-workspace.sh

# 3. Build llama.cpp (CUDA) + whisper.cpp + sherpa-onnx + ONNX Runtime Qwen3-TTS native backend (.build/native/libdeskpet_inference.so).
native_lib="$pipeline_root/.build/native/libdeskpet_inference.so"
needs_native_build=0
if [ ! -f "$native_lib" ]; then
  needs_native_build=1
elif [ "$has_no_build" = "0" ]; then
  if [ -n "$(find "$pipeline_root/native/CMakeLists.txt" "$pipeline_root/native/backend.cpp" "$pipeline_root/native/router.cpp" "$pipeline_root/native/speech.cpp" "$pipeline_root/native/runtime.hpp" "$pipeline_root/native/strict-gpu.cmake" "$pipeline_root/native/router-head-graph.hpp" -newer "$native_lib" 2>/dev/null)" ] || \
     ! grep -q '^DESKPET_TOKENIZERS_CPP:BOOL=ON$' "$pipeline_root/.build/native/CMakeCache.txt" 2>/dev/null; then
    needs_native_build=1
  fi
fi

if [ "$needs_native_build" = "1" ]; then
  echo "[product.sh] Building llama.cpp + ONNX Runtime Qwen3-TTS native backend (.build/native/libdeskpet_inference.so) ..."
  ensure_rust_runtime
  ensure_native_deps
  DESKPET_TOKENIZERS_CPP=1 DESKPET_BUILD_JOBS=2 sh tools/build-native-linux.sh -DDESKPET_PREP=OFF
else
  echo "[product.sh] Using existing native backend: .build/native/libdeskpet_inference.so"
fi

# 4. Build hardware daemon if not already built.
if [ ! -x "$workspace_dir/deskpet-hardware/build/deskpet-hardware" ]; then
  hw_flag=""
  if [ "$has_real" = "1" ] && [ "$has_headless" = "1" ]; then hw_flag="--real-headless"
  elif [ "$has_real" = "1" ]; then hw_flag="--real"
  elif [ "$has_headless" = "1" ]; then hw_flag="--headless"
  fi
  if [ -n "$hw_flag" ]; then
    sh tools/configure-hardware-linux.sh "$workspace_dir/deskpet-hardware" "$workspace_dir/deskpet-hardware/build" "$hw_flag"
  else
    sh tools/configure-hardware-linux.sh "$workspace_dir/deskpet-hardware" "$workspace_dir/deskpet-hardware/build"
  fi
  cmake --build "$workspace_dir/deskpet-hardware/build" --target deskpet-hardware -j 2
else
  echo "[product.sh] Using existing hardware daemon: ../deskpet-hardware/build/deskpet-hardware"
fi

# 5. Prepare harness dependencies if not already installed.
if [ ! -d "$workspace_dir/deskpet-harness/node_modules/tsx" ]; then
  (cd "$workspace_dir/deskpet-harness" && pnpm install --frozen-lockfile)
else
  echo "[product.sh] Using existing harness dependencies: ../deskpet-harness/node_modules"
fi

# 6. Build deskpet-demo if missing or outdated.
export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"

if [ ! -x ".build/debug/deskpet-demo" ] || \
   { [ "$has_no_build" = "0" ] && [ -n "$(find Sources Package.swift -newer ".build/debug/deskpet-demo" 2>/dev/null)" ]; }; then
  swift build --product deskpet-demo -j 2
else
  echo "[product.sh] Using existing deskpet-demo binary: .build/debug/deskpet-demo"
fi

if [ "$has_no_build" = "0" ]; then
  exec .build/debug/deskpet-demo --jetson --no-build "$@"
else
  exec .build/debug/deskpet-demo --jetson "$@"
fi
