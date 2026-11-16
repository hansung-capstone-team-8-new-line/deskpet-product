#!/bin/sh
# Jetson Orin Nano production entry point (dash-compatible /bin/sh).
# - Clones missing repositories (deskpet-pipeline, deskpet-hardware, deskpet-harness) as relative siblings.
# - Bootstraps Swift toolchain/runtime if `swift` is not installed.
# - Builds TensorRT native backend (`libdeskpet_inference.so`) WITHOUT cloning or compiling `llama.cpp`,
#   using official HuggingFace Rust `tokenizers` via `tokenizers-cpp`.
# - Applies INT4 + FP16 quantization (`--int4 --fp16 --memPoolSize=workspace:512M`) for Router backbone & Output model via trtexec (skips if .engine already exists).
# - Supports `--use-audio-gpu` to route STT and TTS through GPUExecutor (`DESKPET_USE_AUDIO_GPU=1`) and configure `stt`/`tts` device as `cuda`.
set -eu

script_dir=$(cd -P "$(dirname "$0")" && pwd)
workspace_dir=$(cd -P "$script_dir/.." && pwd)

use_audio_gpu=0
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
  echo "Usage: $0 [--use-audio-gpu] [deskpet-demo --jetson options...]"
  echo "  --use-audio-gpu   Run STT and TTS via GPUExecutor (DESKPET_USE_AUDIO_GPU=1) with device=cuda"
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
export PATH="$SWIFT_ROOT/usr/bin:$HOME/.local/swift/usr/bin:$HOME/.cargo/bin:$PATH:/usr/src/tensorrt/bin:/usr/local/cuda/bin:/usr/local/bin:$HOME/.local/share/pnpm:$HOME/Library/pnpm"

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

  swift_ver="6.0.3"
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

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SWIFT_ROOT/usr/lib/swift/linux:$PWD/.build/native:$PWD/.build/native/whisper/src:$PWD/.build/native/whisper/ggml/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

export DESKPET_HARNESS_DIR="../deskpet-harness"
export DESKPET_HARDWARE_DIR="../deskpet-hardware"

# 1. Prepare TensorRT engines (INT4+FP16 for router backbone & output model, FP16 for router heads).
#    If .engine file already exists, skip conversion immediately.
TRTEXEC=""
if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC=$(command -v trtexec)
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

if [ -d "Models" ]; then
  for onnx_file in \
    "Models/router/backbone.onnx" \
    "Models/router/model.onnx" \
    "Models/output/model.onnx"; do
    engine_file="${onnx_file%.onnx}.engine"
    if [ -f "$engine_file" ]; then
      echo "[product.sh] Using existing INT4+FP16 TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building INT4+FP16 TensorRT engine with trtexec: $onnx_file -> $engine_file"
      "$TRTEXEC" --onnx="$onnx_file" --saveEngine="$engine_file" --int4 --fp16 --memPoolSize=workspace:512M
    fi
  done

  for onnx_file in \
    "Models/router/heads.onnx"; do
    engine_file="${onnx_file%.onnx}.engine"
    if [ -f "$engine_file" ]; then
      echo "[product.sh] Using existing FP16 TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building FP16 TensorRT engine with trtexec: $onnx_file -> $engine_file"
      "$TRTEXEC" --onnx="$onnx_file" --saveEngine="$engine_file" --fp16 --memPoolSize=workspace:512M
    fi
  done
fi

# 2. Configure Models/native.jetson.local.json via Swift (no Python).
if [ ! -f "Models/native.jetson.local.json" ] && [ -f "Models/native.jetson.example.json" ]; then
  cp "Models/native.jetson.example.json" "Models/native.jetson.local.json"
fi

if [ -f "Models/native.jetson.local.json" ]; then
  if [ "$use_audio_gpu" = "1" ]; then
    swift "$script_dir/configure_jetson_json.swift" --use-audio-gpu
  else
    swift "$script_dir/configure_jetson_json.swift"
  fi
fi

# 3. Prepare lightweight native dependencies (whisper.cpp + official tokenizers-cpp + sherpa-onnx + onnxruntime headers + json.hpp ONLY; NO llama.cpp).
ensure_trt_native_deps() {
  mkdir -p "$pipeline_root/.build/native-src" "$pipeline_root/.build/native-deps/include/nlohmann"

  if [ ! -d "$pipeline_root/.build/native-src/whisper/.git" ]; then
    git clone --no-checkout https://github.com/ggml-org/whisper.cpp.git "$pipeline_root/.build/native-src/whisper"
    git -C "$pipeline_root/.build/native-src/whisper" checkout 231638a0717b7822784297e5980965ffcec05810
  fi

  if [ ! -f "$pipeline_root/.build/native-src/tokenizers-cpp/CMakeLists.txt" ]; then
    git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git "$pipeline_root/.build/native-src/tokenizers-cpp"
  fi

  if [ ! -f "$pipeline_root/.build/native-deps/include/nlohmann/json.hpp" ]; then
    if [ -f "$pipeline_root/.build/native-src/llama/vendor/nlohmann/json.hpp" ]; then
      cp "$pipeline_root/.build/native-src/llama/vendor/nlohmann/json.hpp" "$pipeline_root/.build/native-deps/include/nlohmann/json.hpp"
    else
      curl -fL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -o "$pipeline_root/.build/native-deps/include/nlohmann/json.hpp"
    fi
  fi

  sherpa_mode="cpu"
  if [ "$use_audio_gpu" = "1" ]; then
    sherpa_mode="gpu"
  fi
  cur_sherpa_mode=""
  if [ -f "$pipeline_root/.build/native-deps/sherpa/.mode" ]; then
    cur_sherpa_mode=$(cat "$pipeline_root/.build/native-deps/sherpa/.mode")
  fi
  if [ ! -f "$pipeline_root/.build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so" ] || \
     [ ! -f "$pipeline_root/.build/native-deps/sherpa/include/onnxruntime/core/session/onnxruntime_cxx_api.h" ] || \
     [ "$cur_sherpa_mode" != "$sherpa_mode" ]; then
    sh "$pipeline_root/tools/setup-native-deps-linux.sh"
  fi
}

# 4. Build TensorRT native backend (.build/native/libdeskpet_inference.so) without llama.cpp if missing or outdated.
native_lib="$pipeline_root/.build/native/libdeskpet_inference.so"
if [ ! -f "$native_lib" ] || [ -n "$(find "$script_dir/CMakeLists.txt" "$script_dir/trt_language_model.cpp" "$pipeline_root/native/speech.cpp" -newer "$native_lib" 2>/dev/null)" ]; then
  echo "[product.sh] Building TensorRT native backend (without llama.cpp) ..."
  ensure_rust_runtime
  ensure_trt_native_deps
  sh tools/prepare-cmake-build.sh "$script_dir" "$pipeline_root/.build/native"
  cmake -S "$script_dir" -B "$pipeline_root/.build/native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDESKPET_ROOT="$pipeline_root" \
    -DSHERPA_ROOT="$pipeline_root/.build/native-deps/sherpa"
  cmake --build "$pipeline_root/.build/native" --target deskpet_inference -j 2
else
  echo "[product.sh] Using existing native backend: .build/native/libdeskpet_inference.so"
fi

# 5. Build hardware daemon if not already built.
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

# 6. Prepare harness dependencies if not already installed.
if [ ! -d "$workspace_dir/deskpet-harness/node_modules/tsx" ]; then
  (cd "$workspace_dir/deskpet-harness" && pnpm install --frozen-lockfile)
else
  echo "[product.sh] Using existing harness dependencies: ../deskpet-harness/node_modules"
fi

# 7. Prepare Swift workspace (without touching .build/native TensorRT build) and build deskpet-demo if missing.
mkdir -p .build
stamp=.build/workspace-root
previous_root=""
if [ -f "$stamp" ]; then previous_root=$(cat "$stamp"); fi
if [ "$previous_root" != "$pipeline_root" ]; then
  printf '%s\n' "$pipeline_root" > "$stamp"
fi

export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"

if [ ! -x ".build/debug/deskpet-demo" ]; then
  swift build --product deskpet-demo -j 2
else
  echo "[product.sh] Using existing deskpet-demo binary: .build/debug/deskpet-demo"
fi

if [ "$has_no_build" = "0" ]; then
  exec .build/debug/deskpet-demo --jetson --no-build "$@"
else
  exec .build/debug/deskpet-demo --jetson "$@"
fi
