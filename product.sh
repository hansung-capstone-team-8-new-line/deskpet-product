#!/bin/sh
# Jetson Orin Nano production entry point (dash-compatible /bin/sh).
# - Clones missing repositories (deskpet-pipeline, deskpet-hardware, deskpet-harness) as relative siblings.
# - Bootstraps Swift toolchain/runtime if `swift` is not installed.
# - Builds TensorRT native backend (`libdeskpet_inference.so`) WITHOUT cloning or compiling `llama.cpp`.
# - Applies INT4 quantization (`--int4 --fp16 --memPoolSize=workspace:512M`) for Router backbone & Output model via trtexec (skips if .engine already exists).
set -eu

script_dir=$(cd -P "$(dirname "$0")" && pwd)
workspace_dir=$(cd -P "$script_dir/.." && pwd)

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
export PATH="$SWIFT_ROOT/usr/bin:$HOME/.local/swift/usr/bin:$PATH:/usr/src/tensorrt/bin:/usr/local/cuda/bin:/usr/local/bin:$HOME/.local/share/pnpm:$HOME/Library/pnpm"

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

ensure_swift_runtime

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SWIFT_ROOT/usr/lib/swift/linux:$PWD/.build/native:$PWD/.build/native/whisper/src:$PWD/.build/native/whisper/ggml/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

export DESKPET_HARNESS_DIR="../deskpet-harness"
export DESKPET_HARDWARE_DIR="../deskpet-hardware"

if [ ! -f "Models/native.jetson.local.json" ] && [ -f "Models/native.jetson.example.json" ]; then
  sed 's|\.\./\.\./LLM_test/\.env|../../deskpet-harness/.env|g' "Models/native.jetson.example.json" > "Models/native.jetson.local.json"
elif [ -f "Models/native.jetson.local.json" ]; then
  sed 's|\.\./\.\./LLM_test/\.env|../../deskpet-harness/.env|g' "Models/native.jetson.local.json" > "Models/native.jetson.local.json.tmp"
  mv "Models/native.jetson.local.json.tmp" "Models/native.jetson.local.json"
fi

# 1. Prepare TensorRT engines (INT4 for router backbone & output model, FP16 for router heads).
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
      echo "[product.sh] Using existing INT4 TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building INT4 TensorRT engine with trtexec: $onnx_file -> $engine_file"
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

# 2. Prepare lightweight native dependencies (whisper.cpp + sherpa-onnx + json.hpp ONLY; NO llama.cpp).
ensure_trt_native_deps() {
  mkdir -p "$pipeline_root/.build/native-src" "$pipeline_root/.build/native-deps/include/nlohmann"

  if [ ! -d "$pipeline_root/.build/native-src/whisper/.git" ]; then
    git clone https://github.com/ggml-org/whisper.cpp.git "$pipeline_root/.build/native-src/whisper"
    git -C "$pipeline_root/.build/native-src/whisper" checkout d6bde2632ef7dc2ec2fb93bc2ad6bbd1bf85bd11
  fi

  if [ ! -f "$pipeline_root/.build/native-deps/include/nlohmann/json.hpp" ] && \
     [ ! -f "$pipeline_root/.build/native-src/llama/vendor/nlohmann/json.hpp" ]; then
    curl -fL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
      -o "$pipeline_root/.build/native-deps/include/nlohmann/json.hpp"
  fi

  if [ ! -f "$pipeline_root/.build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so" ]; then
    arch=$(uname -m)
    case "$arch" in
      aarch64|arm64) sherpa_arch="aarch64" ;;
      x86_64|amd64)  sherpa_arch="x64" ;;
      *) echo "[product.sh] Unsupported Linux architecture: $arch" >&2; exit 1 ;;
    esac
    sherpa_archive="sherpa-onnx-v1.12.39-linux-${sherpa_arch}-shared-cpu.tar.bz2"
    curl -fL "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.39/$sherpa_archive" \
      -o "$pipeline_root/.build/native-deps/$sherpa_archive"
    rm -rf "$pipeline_root/.build/native-deps/sherpa"
    mkdir -p "$pipeline_root/.build/native-deps/sherpa"
    tar -xjf "$pipeline_root/.build/native-deps/$sherpa_archive" --strip-components=1 -C "$pipeline_root/.build/native-deps/sherpa"
  fi
}

# 3. Build TensorRT native backend (.build/native/libdeskpet_inference.so) without llama.cpp if not already built.
if [ ! -f "$pipeline_root/.build/native/libdeskpet_inference.so" ]; then
  echo "[product.sh] Building TensorRT native backend (without llama.cpp) ..."
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

# 4. Build hardware daemon if not already built.
if [ ! -x "$workspace_dir/deskpet-hardware/build/deskpet-hardware" ]; then
  hw_flag=""
  has_real=0
  has_headless=0
  for arg in "$@"; do
    if [ "$arg" = "--real-hardware" ]; then has_real=1; fi
    if [ "$arg" = "--headless" ]; then has_headless=1; fi
  done
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

# 6. Prepare Swift workspace (without touching .build/native TensorRT build) and build deskpet-demo if missing.
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

has_no_build=0
for arg in "$@"; do
  if [ "$arg" = "--no-build" ]; then
    has_no_build=1
  fi
done

if [ "$has_no_build" = "0" ]; then
  exec .build/debug/deskpet-demo --jetson --no-build "$@"
else
  exec .build/debug/deskpet-demo --jetson "$@"
fi
