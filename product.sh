#!/bin/sh
# Jetson Orin Nano production entry point (dash-compatible /bin/sh).
# 1. Clones missing repositories (deskpet-pipeline, deskpet-hardware, deskpet-harness) as relative siblings.
# 2. Bootstraps Swift toolchain/runtime if `swift` is not installed.
# 3. Uses pre-built TensorRT .engine / model files immediately if present; builds missing engines via trtexec.
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

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SWIFT_ROOT/usr/lib/swift/linux:$PWD/.build/native:$PWD/.build/native/llama/src:$PWD/.build/native/llama/ggml/src:$PWD/.build/native/llama/ggml/src/ggml-cuda:$PWD/.build/native/whisper/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

export DESKPET_HARNESS_DIR="../deskpet-harness"
export DESKPET_HARDWARE_DIR="../deskpet-hardware"

if [ ! -f "Models/native.jetson.local.json" ] && [ -f "Models/native.jetson.example.json" ]; then
  sed 's|\.\./\.\./LLM_test/\.env|../../deskpet-harness/.env|g' "Models/native.jetson.example.json" > "Models/native.jetson.local.json"
elif [ -f "Models/native.jetson.local.json" ]; then
  sed 's|\.\./\.\./LLM_test/\.env|../../deskpet-harness/.env|g' "Models/native.jetson.local.json" > "Models/native.jetson.local.json.tmp"
  mv "Models/native.jetson.local.json.tmp" "Models/native.jetson.local.json"
fi

# TensorRT engine check & build: if .engine already exists, use it directly without rebuilding.
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
    "Models/output/model.onnx" \
    "Models/emotion/model.onnx" \
    "Models/vad/silero_vad.onnx"; do
    engine_file="${onnx_file%.onnx}.engine"
    if [ -f "$engine_file" ]; then
      echo "[product.sh] Using existing TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building TensorRT engine with trtexec: $onnx_file -> $engine_file"
      "$TRTEXEC" --onnx="$onnx_file" --saveEngine="$engine_file" --fp16
    fi
  done
fi

sh tools/prepare-workspace.sh
export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"

if [ ! -x ".build/debug/deskpet-demo" ]; then
  swift build --product deskpet-demo
else
  echo "[product.sh] Using existing deskpet-demo binary: .build/debug/deskpet-demo"
fi

# If native library, hardware daemon, and harness node_modules are already built, pass --no-build to skip redundant builds.
has_no_build=0
for arg in "$@"; do
  if [ "$arg" = "--no-build" ]; then
    has_no_build=1
  fi
done

if [ "$has_no_build" = "0" ] && \
   [ -f ".build/native/libdeskpet_inference.so" ] && \
   [ -x "../deskpet-hardware/build/deskpet-hardware" ] && \
   [ -d "../deskpet-harness/node_modules/tsx" ]; then
  echo "[product.sh] Pre-built native model library, hardware daemon, and harness dependencies detected; skipping rebuild."
  exec .build/debug/deskpet-demo --jetson --no-build "$@"
fi

exec .build/debug/deskpet-demo --jetson "$@"
