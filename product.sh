#!/bin/sh
# Jetson Orin Nano production entry point (dash-compatible /bin/sh).
# - Clones missing repositories (deskpet-pipeline, deskpet-hardware, deskpet-harness) as relative siblings.
# - Bootstraps Swift toolchain/runtime if `swift` is not installed.
# - Builds TensorRT native backend (`libdeskpet_inference.so`) WITHOUT cloning or compiling `llama.cpp`,
#   using official HuggingFace Rust `tokenizers` via `tokenizers-cpp`.
# - Applies INT4 + FP16 quantization for the Router backbone via trtexec (skips if .engine already exists).
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

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SWIFT_ROOT/usr/lib/swift/linux:$PWD/.build/native-trt:$PWD/.build/native-trt/whisper/src:$PWD/.build/native-trt/whisper/ggml/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"

export DESKPET_HARNESS_DIR="../deskpet-harness"
export DESKPET_HARDWARE_DIR="../deskpet-hardware"

# 1. Prepare TensorRT engines (INT4+FP16 for router backbone, FP16 for router heads).
#    If .engine file already exists, skip conversion immediately.
TRTEXEC=""
if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC=$(command -v trtexec)
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

ensure_router_backbone_onnx() {
  if [ -f "Models/router/backbone.engine" ] || [ -f "Models/router/model.engine" ] || \
     [ -f "Models/router/backbone.onnx" ] || [ -f "Models/router/model.onnx" ]; then
    return 0
  fi
  if [ ! -f "Models/router/base/model.safetensors" ] || [ ! -f "Models/router/backbone/adapter_model.safetensors" ]; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[product.sh] python3 not found; cannot export Models/router/backbone.onnx from base + LoRA." >&2
    return 1
  fi
  echo "[product.sh] Exporting Models/router/backbone.onnx from base + LoRA adapter ..."
  python3 - <<'PY'
import json
import os
import sys

try:
    import torch
    from transformers import AutoConfig, AutoModel
    from peft import PeftModel
except ImportError as exc:
    sys.stderr.write(
        f"[product.sh] Missing Python package for ONNX export ({exc}). "
        "Install torch, transformers, peft, and onnx.\n"
    )
    sys.exit(1)

base_dir = "Models/router/base"
adapter_dir = "Models/router/backbone"
cfg_path = "Models/router/base_config.json" if os.path.isfile("Models/router/base_config.json") else os.path.join(base_dir, "config.json")
out_onnx = "Models/router/backbone.onnx"
tmp_onnx = out_onnx + ".part"

raw_cfg = AutoConfig.from_pretrained(cfg_path)
text_cfg = getattr(raw_cfg, "text_config", raw_cfg)
dtype = torch.float16

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    base_model = Qwen3_5TextModel.from_pretrained(base_dir, config=text_cfg, torch_dtype=dtype, local_files_only=True)
except Exception:
    loaded = AutoModel.from_pretrained(base_dir, config=raw_cfg, torch_dtype=dtype, local_files_only=True)
    base_model = getattr(loaded, "language_model", loaded)

merged = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True).merge_and_unload()
merged.eval().cpu()

class BackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

wrapper = BackboneWrapper(merged)
dummy_ids = torch.ones((1, 8), dtype=torch.int64)
dummy_mask = torch.ones((1, 8), dtype=torch.int64)

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        tmp_onnx,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "last_hidden_state": {1: "seq_len"},
        },
        opset_version=18,
    )

os.replace(tmp_onnx, out_onnx)
print(f"[product.sh] Exported {out_onnx}")
PY
}

if [ -d "Models" ]; then
  ensure_router_backbone_onnx
  for onnx_file in \
    "Models/router/backbone.onnx" \
    "Models/router/model.onnx"; do
    engine_file="${onnx_file%.onnx}.engine"
    if [ -f "$engine_file" ]; then
      echo "[product.sh] Using existing INT4+FP16 TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building INT4+FP16 TensorRT engine with trtexec: $onnx_file -> $engine_file"
      "$TRTEXEC" --onnx="$onnx_file" --saveEngine="$engine_file" --int4 --fp16 \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:1x64,attention_mask:1x64 \
        --maxShapes=input_ids:1x2560,attention_mask:1x2560 \
        --builderOptimizationLevel=0 --maxAuxStreams=0 --memPoolSize=workspace:128M \
        --tacticSources=-CUDNN,-EDGE_MASK_CONVOLUTIONS,-JIT_CONVOLUTIONS
    fi
  done

  for onnx_file in \
    "Models/router/heads.onnx"; do
    engine_file="${onnx_file%.onnx}.engine"
    if [ -f "$engine_file" ]; then
      echo "[product.sh] Using existing FP16 TensorRT engine: $engine_file"
    elif [ -f "$onnx_file" ] && [ -n "$TRTEXEC" ]; then
      echo "[product.sh] Building FP16 TensorRT engine with trtexec: $onnx_file -> $engine_file"
      "$TRTEXEC" --onnx="$onnx_file" --saveEngine="$engine_file" --fp16 \
        --builderOptimizationLevel=0 --maxAuxStreams=0 --memPoolSize=workspace:128M \
        --tacticSources=-CUDNN,-EDGE_MASK_CONVOLUTIONS,-JIT_CONVOLUTIONS
    fi
  done
fi

if [ ! -f "Models/native.jetson.local.json" ] && [ -f "Models/native.jetson.example.json" ]; then
  cp "Models/native.jetson.example.json" "Models/native.jetson.local.json"
fi

if [ -f "Models/native.jetson.local.json" ]; then
  if [ "$use_audio_gpu" = "1" ]; then
    swift "$script_dir/configure_jetson_json.swift" --use-audio-gpu
  else
    swift "$script_dir/configure_jetson_json.swift"
  fi
  swift - <<'SWIFT'
import Foundation
let url = URL(fileURLWithPath: "Models/native.jetson.local.json")
var config = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
var models = config["models"] as! [String: Any]
var output = models["output"] as! [String: Any]
output["backend"] = "llama.cpp"
output["device"] = "cuda"
output["paths"] = ["model": "output/qwen-2b-q4_0.gguf"]
output["library"] = "../.build/native/libdeskpet_inference.so"
models["output"] = output
config["models"] = models
config["library"] = "../.build/native-trt/libdeskpet_inference.so"
let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
try data.write(to: url, options: .atomic)
SWIFT
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
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$nupkg" "$ort_include" $ort_headers <<'PY'
import os, sys, zipfile
nupkg, dest_dir, *headers = sys.argv[1:]
with zipfile.ZipFile(nupkg) as zf:
    for h in headers:
        with open(os.path.join(dest_dir, h), "wb") as out:
            out.write(zf.read(f"build/native/include/{h}"))
PY
  else
    echo "[product.sh] Neither unzip nor python3 is available to extract onnxruntime headers from $nupkg" >&2
    exit 1
  fi
  rm -f "$nupkg"
}

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
    if ! sh "$pipeline_root/tools/setup-native-deps-linux.sh"; then
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

# 4. Build TensorRT native backend separately without replacing the existing llama.cpp backend.
native_lib="$pipeline_root/.build/native-trt/libdeskpet_inference.so"
if [ ! -f "$native_lib" ] || [ -n "$(find "$script_dir/CMakeLists.txt" "$script_dir/trt_language_model.cpp" "$pipeline_root/native/speech.cpp" -newer "$native_lib" 2>/dev/null)" ]; then
  echo "[product.sh] Building TensorRT native backend (without llama.cpp) ..."
  ensure_rust_runtime
  ensure_trt_native_deps
  sh tools/prepare-cmake-build.sh "$script_dir" "$pipeline_root/.build/native-trt"
  cmake -S "$script_dir" -B "$pipeline_root/.build/native-trt" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDESKPET_ROOT="$pipeline_root" \
    -DSHERPA_ROOT="$pipeline_root/.build/native-deps/sherpa"
  cmake --build "$pipeline_root/.build/native-trt" --target deskpet_inference -j 2
else
  echo "[product.sh] Using existing native backend: .build/native-trt/libdeskpet_inference.so"
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
