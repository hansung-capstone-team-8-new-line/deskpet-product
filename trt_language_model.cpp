#include "runtime.hpp"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(DESKPET_HAS_TOKENIZERS_CPP)
#include <tokenizers_cpp.h>
#endif

#if defined(DESKPET_HAS_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace {

std::pair<int, double> best_logit(const std::vector<float> &v) {
  require(!v.empty(), "EMPTY_LOGITS");
  int i = static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
  double sum = 0.0;
  for (float f : v) sum += std::exp(static_cast<double>(f - v[i]));
  return {i, 1.0 / sum};
}

size_t unicode_scalars(const std::string &s) {
  size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0) != 0x80) ++n;
  }
  return n;
}

size_t scalar_to_byte(const std::string &s, size_t c) {
  size_t n = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if ((static_cast<uint8_t>(s[i]) & 0xC0) != 0x80 && n++ == c) return i;
  }
  return s.size();
}

std::pair<size_t, size_t> trimmed_bounds(const std::string &s) {
  auto white = [](unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
  };
  size_t a = 0, b = s.size();
  while (a < b && white(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && white(static_cast<unsigned char>(s[b - 1]))) --b;
  return {a, b};
}

std::string trim_copy(const std::string &s) {
  auto [a, b] = trimmed_bounds(s);
  return s.substr(a, b - a);
}

bool file_exists(const std::string &path) {
  if (path.empty()) return false;
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

bool file_mtime_ge(const std::string &candidate, const std::string &reference) {
  struct stat st_c{}, st_r{};
  if (stat(candidate.c_str(), &st_c) != 0 || st_c.st_size <= 0) return false;
  if (stat(reference.c_str(), &st_r) != 0) return true;
  return st_c.st_mtime >= st_r.st_mtime;
}

std::string dirname_of(const std::string &path) {
  auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0, pos);
}

std::string opt_path(const json &c, const char *key) {
  if (!c.contains("paths") || !c["paths"].is_object()) return {};
  const auto &p = c["paths"];
  if (!p.contains(key) || !p[key].is_string()) return {};
  return p[key].get<std::string>();
}

std::string strip_think_prefix(const std::string &content) {
  size_t end_tag = content.find("</think>");
  if (end_tag == std::string::npos) return content;
  size_t pos = end_tag + 8;
  while (pos < content.size() && content[pos] == '\n') ++pos;
  return content.substr(pos);
}

std::string render_messages(const json &messages, bool is_router) {
  require(messages.is_array() && !messages.empty(), "CHAT_MESSAGES_EMPTY");
  if (is_router) {
    for (const auto &m : messages) {
      std::string role = m.at("role").get<std::string>();
      require(role == "user" || role == "assistant", "ROUTER_CONTEXT_ROLE_UNSUPPORTED");
    }
    require(messages.back().at("role").get<std::string>() == "user", "ROUTER_CURRENT_MESSAGE_REQUIRED");
  }

  std::string out;
  size_t start_idx = 0;
  if (!is_router && messages[0].at("role").get<std::string>() == "system") {
    std::string sys = trim_copy(messages[0].at("content").get<std::string>());
    out += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    start_idx = 1;
  }
  for (size_t i = start_idx; i < messages.size(); ++i) {
    std::string role = messages[i].at("role").get<std::string>();
    std::string content = trim_copy(messages[i].at("content").get<std::string>());
    if (role == "system") {
      require(false, "SYSTEM_MESSAGE_MUST_BE_FIRST");
    } else if (role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (role == "assistant") {
      content = trim_copy(strip_think_prefix(content));
      out += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
    } else {
      require(false, "UNEXPECTED_MESSAGE_ROLE");
    }
  }
  out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  return out;
}

// Official HuggingFace Rust tokenizers wrapper via mlc-ai/tokenizers-cpp.
struct OfficialTokenizer {
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
  std::unique_ptr<tokenizers::Tokenizer> impl;
#endif
  std::set<int32_t> added_token_ids;
  std::set<int32_t> special_ids;
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
  int32_t im_end_id = -1;
  int32_t eot_id = -1;

  void init_unicode_to_byte() {
    bool in_bs[256] = {};
    auto mark = [&](int a, int b) {
      for (int i = a; i <= b; ++i) in_bs[i] = true;
    };
    mark(33, 126);
    mark(161, 172);
    mark(174, 255);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp = in_bs[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
      unicode_to_byte[cp] = static_cast<uint8_t>(b);
    }
  }

  static uint32_t next_cp(const std::string &s, size_t &pos) {
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if ((c & 0x80) == 0) {
      pos += 1;
      return c;
    }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
      uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
      pos += 2;
      return cp;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
      uint32_t cp = ((c & 0x0F) << 12) |
                    ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
                    (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
      pos += 3;
      return cp;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
      uint32_t cp = ((c & 0x07) << 18) |
                    ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
                    ((static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) |
                    (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
      pos += 4;
      return cp;
    }
    pos += 1;
    return c;
  }

  void load(const std::string &path) {
    init_unicode_to_byte();
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "TOKENIZER_NOT_FOUND: " + path);
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    require(!blob.empty(), "TOKENIZER_EMPTY: " + path);

    json j = json::parse(blob, nullptr, false);
    require(!j.is_discarded(), "TOKENIZER_INVALID_JSON: " + path);
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
      for (const auto &at : j["added_tokens"]) {
        if (at.contains("id") && at.contains("content")) {
          int32_t id = at["id"].get<int32_t>();
          added_token_ids.insert(id);
          if (at.value("special", false)) special_ids.insert(id);
        }
      }
    }

#if defined(DESKPET_HAS_TOKENIZERS_CPP)
    impl = tokenizers::Tokenizer::FromBlobJSON(blob);
    require(bool(impl), "TOKENIZER_INIT_FAILED: " + path);
    im_end_id = impl->TokenToId("<|im_end|>");
    eot_id = impl->TokenToId("<|endoftext|>");
    if (im_end_id >= 0) special_ids.insert(im_end_id);
    if (eot_id >= 0) special_ids.insert(eot_id);
#else
    require(false, "TOKENIZERS_CPP_NOT_AVAILABLE");
#endif
  }

  std::vector<int32_t> encode(const std::string &text) const {
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
    return impl->Encode(text);
#else
    (void)text;
    return {};
#endif
  }

  std::string decode(const std::vector<int32_t> &ids) const {
    if (ids.empty()) return {};
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
    std::vector<int32_t> filtered;
    filtered.reserve(ids.size());
    for (int32_t id : ids) {
      if (special_ids.find(id) == special_ids.end()) filtered.push_back(id);
    }
    return filtered.empty() ? std::string() : impl->Decode(filtered);
#else
    return {};
#endif
  }

  bool is_eog(int32_t id) const {
    return (im_end_id >= 0 && id == im_end_id) || (eot_id >= 0 && id == eot_id);
  }

  // Raw byte representation for a single token ID (including partial UTF-8 byte tokens) for span alignment.
  std::string raw_piece(int32_t id) const {
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
    std::string tok = impl->IdToToken(id);
    if (added_token_ids.find(id) != added_token_ids.end()) return tok;
    std::string bytes;
    bytes.reserve(tok.size());
    for (size_t pos = 0; pos < tok.size();) {
      uint32_t cp = next_cp(tok, pos);
      auto it = unicode_to_byte.find(cp);
      if (it != unicode_to_byte.end()) {
        bytes.push_back(static_cast<char>(it->second));
      }
    }
    return bytes;
#else
    (void)id;
    return {};
#endif
  }
};

#if defined(DESKPET_HAS_TENSORRT)

inline void check_cuda(cudaError_t err, const char *what) {
  require(err == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(err));
}

class TrtLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }
} g_trt_logger;

struct TrtDeleter {
  template <typename T>
  void operator()(T *obj) const noexcept { delete obj; }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

std::vector<char> read_binary_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  require(bool(f), "TRT_ENGINE_READ_FAILED: " + path);
  auto sz = f.tellg();
  require(sz > 0, "TRT_ENGINE_EMPTY: " + path);
  std::vector<char> buf(static_cast<size_t>(sz));
  f.seekg(0, std::ios::beg);
  f.read(buf.data(), sz);
  require(bool(f), "TRT_ENGINE_READ_TRUNCATED: " + path);
  return buf;
}

void write_binary_file(const std::string &path, const void *data, size_t size) {
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!f) return;
  }
  std::rename(tmp.c_str(), path.c_str());
}

size_t dtype_bytes(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 4;
  }
}

struct DeviceBuffer {
  void *ptr = nullptr;
  size_t capacity = 0;

  ~DeviceBuffer() { reset(); }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr), capacity(o.capacity) {
    o.ptr = nullptr;
    o.capacity = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
    if (this != &o) {
      reset();
      ptr = o.ptr;
      capacity = o.capacity;
      o.ptr = nullptr;
      o.capacity = 0;
    }
    return *this;
  }

  void ensure(size_t bytes) {
    if (bytes <= capacity && ptr) return;
    reset();
    size_t alloc = std::max<size_t>(bytes, 256);
    check_cuda(cudaMalloc(&ptr, alloc), "cudaMalloc DeviceBuffer");
    capacity = alloc;
  }

  void reset() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      capacity = 0;
    }
  }
};

// Generic TensorRT engine wrapper for GPU sub-engines (heads, argmax) using TensorRT built-in layers.
struct TrtSubEngine {
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> context;
  std::unordered_map<std::string, DeviceBuffer> in_buffers;
  std::unordered_map<std::string, DeviceBuffer> out_buffers;

  void init_from_plan(nvinfer1::IRuntime *rt, const void *data, size_t size) {
    engine.reset(rt->deserializeCudaEngine(data, size));
    require(bool(engine), "TRT_SUBENGINE_INIT_FAILED");
    context.reset(engine->createExecutionContext());
    require(bool(context), "TRT_SUBCONTEXT_INIT_FAILED");
  }

  void run(cudaStream_t stream,
           const std::vector<std::pair<std::string, std::pair<const void *, nvinfer1::Dims>>> &device_inputs,
           const std::vector<std::pair<std::string, std::vector<float> *>> &float_outputs,
           const std::vector<std::pair<std::string, int32_t *>> &int32_outputs = {},
           const std::vector<std::pair<std::string, std::pair<const float *, nvinfer1::Dims>>> &host_float_inputs = {}) {
    for (const auto &in : host_float_inputs) {
      size_t elems = 1;
      for (int d = 0; d < in.second.second.nbDims; ++d) elems *= static_cast<size_t>(in.second.second.d[d]);
      size_t bytes = elems * sizeof(float);
      auto &buf = in_buffers[in.first];
      buf.ensure(bytes);
      check_cuda(cudaMemcpyAsync(buf.ptr, in.second.first, bytes, cudaMemcpyHostToDevice, stream),
                 "TRT_SUB_H2D_FLOAT");
      auto dims = engine->getTensorShape(in.first.c_str());
      for (int d = 0; d < dims.nbDims; ++d) {
        if (dims.d[d] == -1) {
          require(context->setInputShape(in.first.c_str(), in.second.second),
                  "TRT_SUB_SET_SHAPE_FAILED: " + in.first);
          break;
        }
      }
      require(context->setTensorAddress(in.first.c_str(), buf.ptr),
              "TRT_SUB_SET_HOST_IN_ADDR_FAILED: " + in.first);
    }

    for (const auto &in : device_inputs) {
      auto dims = engine->getTensorShape(in.first.c_str());
      for (int d = 0; d < dims.nbDims; ++d) {
        if (dims.d[d] == -1) {
          require(context->setInputShape(in.first.c_str(), in.second.second),
                  "TRT_SUB_SET_SHAPE_FAILED: " + in.first);
          break;
        }
      }
      require(context->setTensorAddress(in.first.c_str(), const_cast<void *>(in.second.first)),
              "TRT_SUB_SET_IN_ADDR_FAILED: " + in.first);
    }

    int nb_io = engine->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
      const char *name = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;
      auto dims = context->getTensorShape(name);
      auto dt = engine->getTensorDataType(name);
      size_t elems = 1;
      for (int d = 0; d < dims.nbDims; ++d) {
        require(dims.d[d] > 0, std::string("TRT_SUB_OUT_DIM_UNRESOLVED: ") + name);
        elems *= static_cast<size_t>(dims.d[d]);
      }
      auto &buf = out_buffers[name];
      buf.ensure(elems * dtype_bytes(dt));
      require(context->setTensorAddress(name, buf.ptr), std::string("TRT_SUB_SET_OUT_ADDR_FAILED: ") + name);
    }

    require(context->enqueueV3(stream), "TRT_SUB_ENQUEUE_FAILED");

    for (const auto &out : float_outputs) {
      auto dims = context->getTensorShape(out.first.c_str());
      size_t elems = 1;
      for (int d = 0; d < dims.nbDims; ++d) elems *= static_cast<size_t>(dims.d[d]);
      out.second->resize(elems);
      check_cuda(cudaMemcpyAsync(out.second->data(), out_buffers.at(out.first).ptr,
                                 elems * sizeof(float), cudaMemcpyDeviceToHost, stream),
                 "TRT_SUB_MEMCPY_FLOAT");
    }
    for (const auto &out : int32_outputs) {
      check_cuda(cudaMemcpyAsync(out.second, out_buffers.at(out.first).ptr,
                                 sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                 "TRT_SUB_MEMCPY_INT32");
    }
    check_cuda(cudaStreamSynchronize(stream), "TRT_SUB_SYNC");
  }
};

struct TensorRtEngineRunner {
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;

  struct IoBinding {
    std::string name;
    nvinfer1::TensorIOMode mode;
    nvinfer1::DataType dtype;
    nvinfer1::Dims static_dims;
    DeviceBuffer buf;
  };
  std::vector<IoBinding> io_bindings;
  int primary_output_index = -1;
  int output_dim = 0;
  nvinfer1::DataType primary_output_dtype = nvinfer1::DataType::kFLOAT;
  std::unique_ptr<TrtSubEngine> argmax_engine;

  ~TensorRtEngineRunner() {
    argmax_engine.reset();
    io_bindings.clear();
    context.reset();
    engine.reset();
    runtime.reset();
    if (stream) cudaStreamDestroy(stream);
  }

  void build_argmax_subengine(int vocab_size, nvinfer1::DataType dt) {
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_trt_logger));
    require(bool(builder), "TRT_ARGMAX_BUILDER_FAILED");
    TrtUniquePtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto *in = net->addInput("logits", dt, nvinfer1::Dims2{1, vocab_size});
    require(in != nullptr, "TRT_ARGMAX_INPUT_FAILED");
    auto *topk = net->addTopK(*in, nvinfer1::TopKOperation::kMAX, 1, 1U << 1);
    require(topk != nullptr, "TRT_ARGMAX_TOPK_FAILED");
    auto *idx = topk->getOutput(1);
    idx->setName("token");
    net->markOutput(*idx);

    TrtUniquePtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64ULL << 20);
    TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
    require(bool(plan), "TRT_ARGMAX_BUILD_FAILED");
    argmax_engine = std::make_unique<TrtSubEngine>();
    argmax_engine->init_from_plan(runtime.get(), plan->data(), plan->size());
  }

  void load_or_build(const std::string &engine_path, const std::string &onnx_path, int max_seq, bool is_router) {
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
    require(bool(runtime), "TRT_RUNTIME_FAILED");

    if (file_exists(engine_path)) {
      auto bytes = read_binary_file(engine_path);
      engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    } else if (file_exists(onnx_path)) {
      TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_trt_logger));
      require(bool(builder), "TRT_BUILDER_FAILED");
      TrtUniquePtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(
          1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
      TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_trt_logger));
      require(parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)),
              "TRT_ONNX_PARSE_FAILED: " + onnx_path);
      TrtUniquePtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
      // Enforce INT4 + FP16 and cap workspace memory to 512MB for Jetson Orin Nano unified memory safety.
      cfg->setFlag(nvinfer1::BuilderFlag::kINT4);
      cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
      cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL << 20);
      auto *profile = builder->createOptimizationProfile();
      for (int i = 0; i < net->getNbInputs(); ++i) {
        auto *in = net->getInput(i);
        auto dims = in->getDimensions();
        if (dims.nbDims == 2) {
          profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 1});
          profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{1, std::min(64, max_seq)});
          profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{1, max_seq});
        }
      }
      cfg->addOptimizationProfile(profile);
      TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
      require(bool(plan), "TRT_BUILD_FAILED: " + onnx_path);
      if (!engine_path.empty()) write_binary_file(engine_path, plan->data(), plan->size());
      engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    } else {
      require(false, "TRT_MODEL_MISSING: " + engine_path);
    }

    require(bool(engine), "TRT_ENGINE_INIT_FAILED: " + engine_path);
    context.reset(engine->createExecutionContext());
    require(bool(context), "TRT_CONTEXT_INIT_FAILED: " + engine_path);

    int nb_io = engine->getNbIOTensors();
    int first_out = -1;
    int named_out = -1;
    bool has_input = false;
    for (int i = 0; i < nb_io; ++i) {
      const char *name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      auto dt = engine->getTensorDataType(name);
      auto dims = engine->getTensorShape(name);
      IoBinding b;
      b.name = name;
      b.mode = mode;
      b.dtype = dt;
      b.static_dims = dims;
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        has_input = true;
        std::string lower = b.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool known = (lower.find("input") != std::string::npos ||
                      lower.find("mask") != std::string::npos ||
                      lower.find("position") != std::string::npos ||
                      lower == "ids" || lower == "tokens");
        require(known && dims.nbDims == 2, "TRT_UNSUPPORTED_INPUT_TENSOR: " + b.name);
        b.buf.ensure(static_cast<size_t>(max_seq) * dtype_bytes(dt));
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        if (first_out < 0) first_out = static_cast<int>(io_bindings.size());
        std::string lower = b.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (is_router && (lower.find("hidden") != std::string::npos)) {
          named_out = static_cast<int>(io_bindings.size());
        } else if (!is_router && (lower.find("logit") != std::string::npos)) {
          named_out = static_cast<int>(io_bindings.size());
        }
      }
      io_bindings.push_back(std::move(b));
    }
    primary_output_index = (named_out >= 0) ? named_out : first_out;
    require(has_input && primary_output_index >= 0, "TRT_IO_TENSOR_MISSING: " + engine_path);
    const auto &pout = io_bindings[primary_output_index];
    primary_output_dtype = pout.dtype;
    if (pout.static_dims.nbDims >= 1 && pout.static_dims.d[pout.static_dims.nbDims - 1] > 0) {
      output_dim = static_cast<int>(pout.static_dims.d[pout.static_dims.nbDims - 1]);
    }
    if (!is_router && output_dim > 0) {
      build_argmax_subengine(output_dim, primary_output_dtype);
    }
  }

  struct GpuOutputView {
    const void *device_ptr = nullptr;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    int seq_dim = 0;
    int feature_dim = 0;
  };

  GpuOutputView execute_sequence_gpu(const std::vector<int32_t> &tokens) {
    int seq_len = static_cast<int>(tokens.size());
    require(seq_len > 0, "TRT_EMPTY_SEQUENCE");

    for (auto &b : io_bindings) {
      if (b.mode != nvinfer1::TensorIOMode::kINPUT) continue;
      if (b.static_dims.d[1] == -1 || b.static_dims.d[0] == -1) {
        require(context->setInputShape(b.name.c_str(), nvinfer1::Dims2{1, seq_len}),
                "TRT_SET_INPUT_SHAPE_FAILED: " + b.name);
      } else {
        require(b.static_dims.d[1] == seq_len, "TRT_STATIC_INPUT_LENGTH_MISMATCH: " + b.name);
      }

      std::string lower = b.name;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      bool is_mask = lower.find("mask") != std::string::npos;
      bool is_pos = lower.find("position") != std::string::npos;

      size_t bytes = static_cast<size_t>(seq_len) * dtype_bytes(b.dtype);
      b.buf.ensure(bytes);
      if (b.dtype == nvinfer1::DataType::kINT64) {
        std::vector<int64_t> host(seq_len);
        if (is_mask) std::fill(host.begin(), host.end(), 1);
        else if (is_pos) for (int i = 0; i < seq_len; ++i) host[i] = i;
        else for (int i = 0; i < seq_len; ++i) host[i] = tokens[i];
        check_cuda(cudaMemcpyAsync(b.buf.ptr, host.data(), bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync input64");
      } else {
        std::vector<int32_t> host(seq_len);
        if (is_mask) std::fill(host.begin(), host.end(), 1);
        else if (is_pos) for (int i = 0; i < seq_len; ++i) host[i] = i;
        else host = tokens;
        check_cuda(cudaMemcpyAsync(b.buf.ptr, host.data(), bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync input32");
      }
      require(context->setTensorAddress(b.name.c_str(), b.buf.ptr),
              "TRT_SET_INPUT_ADDR_FAILED: " + b.name);
    }

    int resolved_seq = seq_len;
    int resolved_feat = output_dim;
    for (size_t idx = 0; idx < io_bindings.size(); ++idx) {
      auto &b = io_bindings[idx];
      if (b.mode != nvinfer1::TensorIOMode::kOUTPUT) continue;
      auto dims = context->getTensorShape(b.name.c_str());
      require(dims.nbDims >= 2, "TRT_OUTPUT_RANK_INVALID: " + b.name);
      size_t elems = 1;
      for (int d = 0; d < dims.nbDims; ++d) {
        require(dims.d[d] > 0, "TRT_OUTPUT_DIM_UNRESOLVED: " + b.name);
        elems *= static_cast<size_t>(dims.d[d]);
      }
      b.buf.ensure(elems * dtype_bytes(b.dtype));
      require(context->setTensorAddress(b.name.c_str(), b.buf.ptr),
              "TRT_SET_OUTPUT_ADDR_FAILED: " + b.name);
      if (static_cast<int>(idx) == primary_output_index) {
        resolved_feat = static_cast<int>(dims.d[dims.nbDims - 1]);
        resolved_seq = (dims.nbDims >= 3) ? static_cast<int>(dims.d[dims.nbDims - 2])
                                          : static_cast<int>(dims.d[0]);
      }
    }
    output_dim = resolved_feat;

    require(context->enqueueV3(stream), "TRT_ENQUEUE_FAILED");

    const auto &pout = io_bindings[primary_output_index];
    return {pout.buf.ptr, pout.dtype, resolved_seq, resolved_feat};
  }

  // Runs TensorRT output model + TensorRT TopK(k=1) argmax entirely on GPU; returns 1 int32 token.
  int32_t forward_next_token_gpu(const std::vector<int32_t> &tokens) {
    auto view = execute_sequence_gpu(tokens);
    require(view.feature_dim > 0 && view.seq_dim > 0, "TRT_LOGITS_DIM_INVALID");
    if (!argmax_engine) {
      build_argmax_subengine(view.feature_dim, view.dtype);
    }
    size_t last_byte_offset =
        static_cast<size_t>(view.seq_dim - 1) * static_cast<size_t>(view.feature_dim) * dtype_bytes(view.dtype);
    const void *d_last_logits = static_cast<const char *>(view.device_ptr) + last_byte_offset;
    int32_t next_tok = 0;
    argmax_engine->run(stream,
                       {{"logits", {d_last_logits, nvinfer1::Dims2{1, view.feature_dim}}}},
                       {},
                       {{"token", &next_tok}});
    return next_tok;
  }
};

// Builds & caches pure TensorRT FP16 engines from heads-f16.gguf using TensorRT built-in layers
// (MatrixMultiply, ElementWise, Unary ERF, Activation, SoftMax, TopK). Zero CPU inference, zero custom CUDA kernels.
struct TrtHeadEngines {
  nvinfer1::IRuntime *runtime = nullptr;
  cudaStream_t stream = nullptr;
  nvinfer1::DataType backbone_dt = nvinfer1::DataType::kHALF;
  int hidden = 1024;
  int vocab_size = 248320;
  int max_seq = 2560;

  TrtSubEngine classifiers_engine;
  TrtSubEngine confidence_engine;
  TrtSubEngine spans_engine;
  TrtSubEngine gen_task_engine;
  TrtSubEngine gen_note_engine;

  struct GgufReader {
    ggml_context *ctx = nullptr;
    gguf_context *meta = nullptr;
    std::string path;

    explicit GgufReader(const std::string &file) : path(file) {
      gguf_init_params params{true, &ctx};
      meta = gguf_init_from_file(file.c_str(), params);
      require(meta && ctx, "INVALID_HEADS_GGUF");
      for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        auto *t = ggml_get_tensor(ctx, gguf_get_tensor_name(meta, i));
        require(t && t->type == GGML_TYPE_F16, "HEADS_MUST_BE_FP16");
      }
    }
    ~GgufReader() {
      if (meta) gguf_free(meta);
      if (ctx) ggml_free(ctx);
    }

    const ggml_tensor *tensor_info(const std::string &name) const {
      auto *t = ggml_get_tensor(ctx, name.c_str());
      require(t != nullptr, "MISSING_HEAD: " + name);
      return t;
    }

    std::vector<uint16_t> read_f16(const std::string &name) const {
      auto idx = gguf_find_tensor(meta, name.c_str());
      require(idx >= 0, "MISSING_HEAD: " + name);
      auto *t = ggml_get_tensor(ctx, name.c_str());
      size_t bytes = ggml_nbytes(t);
      std::vector<uint16_t> data(bytes / sizeof(uint16_t));
      std::ifstream in(path, std::ios::binary);
      require(bool(in), "HEAD_FILE_OPEN_FAILED: " + path);
      in.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, idx));
      in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
      require(static_cast<size_t>(in.gcount()) == bytes, "TRUNCATED_HEADS: " + name);
      return data;
    }
  };

  struct PinnedBuffers {
    std::vector<std::vector<uint16_t>> f16;
    std::vector<std::vector<float>> f32;
  };

  static nvinfer1::ITensor *ensure_dtype(nvinfer1::INetworkDefinition *net, nvinfer1::ITensor *t,
                                         nvinfer1::DataType target) {
    if (t->getType() == target) return t;
    auto *id = net->addIdentity(*t);
    require(id != nullptr, "TRT_IDENTITY_CAST_FAILED");
    id->setOutputType(0, target);
    return id->getOutput(0);
  }

  static nvinfer1::ITensor *add_linear_f16(nvinfer1::INetworkDefinition *net,
                                           nvinfer1::ITensor *x_f16,
                                           const GgufReader &gguf,
                                           const std::string &prefix,
                                           bool has_bias,
                                           PinnedBuffers &pinned) {
    const auto *wt = gguf.tensor_info(prefix + ".weight");
    int in_dim = static_cast<int>(wt->ne[0]);
    int out_dim = static_cast<int>(wt->ne[1]);
    pinned.f16.push_back(gguf.read_f16(prefix + ".weight"));
    const auto &w_buf = pinned.f16.back();
    auto *w_layer = net->addConstant(
        nvinfer1::Dims2{out_dim, in_dim},
        nvinfer1::Weights{nvinfer1::DataType::kHALF, w_buf.data(), static_cast<int64_t>(w_buf.size())});
    require(w_layer != nullptr, "TRT_CONST_WEIGHT_FAILED: " + prefix);
    auto *mm = net->addMatrixMultiply(*x_f16, nvinfer1::MatrixOperation::kNONE,
                                      *w_layer->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    require(mm != nullptr, "TRT_MATMUL_FAILED: " + prefix);
    nvinfer1::ITensor *cur = mm->getOutput(0);
    if (has_bias) {
      const auto *bt = gguf.tensor_info(prefix + ".bias");
      require(static_cast<int>(bt->ne[0]) == out_dim, "HEAD_BIAS_DIM_MISMATCH: " + prefix);
      pinned.f16.push_back(gguf.read_f16(prefix + ".bias"));
      const auto &b_buf = pinned.f16.back();
      auto *b_layer = net->addConstant(
          nvinfer1::Dims2{1, out_dim},
          nvinfer1::Weights{nvinfer1::DataType::kHALF, b_buf.data(), static_cast<int64_t>(b_buf.size())});
      require(b_layer != nullptr, "TRT_CONST_BIAS_FAILED: " + prefix);
      auto *sum = net->addElementWise(*cur, *b_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
      require(sum != nullptr, "TRT_BIAS_ADD_FAILED: " + prefix);
      cur = sum->getOutput(0);
    }
    return cur;
  }

  static std::vector<float> f16_to_f32_vec(const std::vector<uint16_t> &src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      __half h;
      std::memcpy(&h, &src[i], sizeof(uint16_t));
      dst[i] = __half2float(h);
    }
    return dst;
  }

  void load_or_build_subengine(
      TrtSubEngine &target,
      const std::string &cache_path,
      const std::string &gguf_path,
      const std::function<void(nvinfer1::IBuilder *, nvinfer1::INetworkDefinition *,
                               nvinfer1::IBuilderConfig *, PinnedBuffers &)> &build_fn) {
    if (file_mtime_ge(cache_path, gguf_path)) {
      auto bytes = read_binary_file(cache_path);
      target.init_from_plan(runtime, bytes.data(), bytes.size());
      return;
    }
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_trt_logger));
    require(bool(builder), "TRT_HEAD_BUILDER_FAILED");
    TrtUniquePtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    TrtUniquePtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);
    PinnedBuffers pinned;
    build_fn(builder.get(), net.get(), cfg.get(), pinned);
    TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
    require(bool(plan), "TRT_HEAD_BUILD_FAILED: " + cache_path);
    write_binary_file(cache_path, plan->data(), plan->size());
    target.init_from_plan(runtime, plan->data(), plan->size());
  }

  void init(const std::string &gguf_path, nvinfer1::IRuntime *rt, cudaStream_t s,
            nvinfer1::DataType b_dt, int h_dim, int m_seq) {
    runtime = rt;
    stream = s;
    backbone_dt = b_dt;
    hidden = h_dim;
    max_seq = m_seq;

    std::string dt_tag = (backbone_dt == nvinfer1::DataType::kHALF) ? "f16" : "f32";
    GgufReader gguf(gguf_path);
    vocab_size = static_cast<int>(gguf.tensor_info("generators.Task.weight")->ne[1]);

    // 1. Classifiers engine: Intention, Operation, Priority
    load_or_build_subengine(
        classifiers_engine,
        gguf_path + ".classifiers." + dt_tag + ".engine",
        gguf_path,
        [&](nvinfer1::IBuilder *, nvinfer1::INetworkDefinition *net, nvinfer1::IBuilderConfig *, PinnedBuffers &pinned) {
          auto *in = net->addInput("pooled", backbone_dt, nvinfer1::Dims2{1, hidden});
          auto *x = ensure_dtype(net, in, nvinfer1::DataType::kHALF);
          for (const char *name : {"Intention", "Operation", "Priority"}) {
            auto *y = add_linear_f16(net, x, gguf, std::string("classifiers.") + name, true, pinned);
            auto *out = ensure_dtype(net, y, nvinfer1::DataType::kFLOAT);
            out->setName(name);
            net->markOutput(*out);
          }
        });

    // 2. Confidence MLP engine: Linear(9->16) + GELU(ERF) + Linear(16->1) + Sigmoid
    load_or_build_subengine(
        confidence_engine,
        gguf_path + ".confidence.engine",
        gguf_path,
        [&](nvinfer1::IBuilder *, nvinfer1::INetworkDefinition *net, nvinfer1::IBuilderConfig *, PinnedBuffers &pinned) {
          auto *in = net->addInput("features", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, 9});
          auto *x_f16 = ensure_dtype(net, in, nvinfer1::DataType::kHALF);
          auto *h_f16 = add_linear_f16(net, x_f16, gguf, "confidence.0", true, pinned);
          auto *h = ensure_dtype(net, h_f16, nvinfer1::DataType::kFLOAT);

          pinned.f32.push_back({0.7071067811865475f});
          pinned.f32.push_back({1.0f});
          pinned.f32.push_back({0.5f});
          auto *c_inv_sqrt2 = net->addConstant(nvinfer1::Dims2{1, 1},
              nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pinned.f32[0].data(), 1});
          auto *c_one = net->addConstant(nvinfer1::Dims2{1, 1},
              nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pinned.f32[1].data(), 1});
          auto *c_half = net->addConstant(nvinfer1::Dims2{1, 1},
              nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pinned.f32[2].data(), 1});

          auto *scaled = net->addElementWise(*h, *c_inv_sqrt2->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
          auto *erf_l = net->addUnary(*scaled->getOutput(0), nvinfer1::UnaryOperation::kERF);
          auto *one_plus = net->addElementWise(*erf_l->getOutput(0), *c_one->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
          auto *half_h = net->addElementWise(*h, *c_half->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
          auto *gelu = net->addElementWise(*half_h->getOutput(0), *one_plus->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

          const auto *w2t = gguf.tensor_info("confidence.2.weight");
          int h_dim2 = static_cast<int>(w2t->ne[0]);
          pinned.f32.push_back(f16_to_f32_vec(gguf.read_f16("confidence.2.weight")));
          pinned.f32.push_back(f16_to_f32_vec(gguf.read_f16("confidence.2.bias")));
          auto *w2 = net->addConstant(nvinfer1::Dims2{1, h_dim2},
              nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pinned.f32[3].data(), h_dim2});
          auto *b2 = net->addConstant(nvinfer1::Dims2{1, 1},
              nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pinned.f32[4].data(), 1});
          auto *mm2 = net->addMatrixMultiply(*gelu->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                             *w2->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
          auto *sum2 = net->addElementWise(*mm2->getOutput(0), *b2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
          auto *sig = net->addActivation(*sum2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
          sig->getOutput(0)->setName("score");
          net->markOutput(*sig->getOutput(0));
        });

    // 3. Spans engine: W * [s_t; last] + b == (W_left * s_t) + (W_right * last + b)
    load_or_build_subengine(
        spans_engine,
        gguf_path + ".spans." + dt_tag + ".engine",
        gguf_path,
        [&](nvinfer1::IBuilder *builder, nvinfer1::INetworkDefinition *net, nvinfer1::IBuilderConfig *cfg, PinnedBuffers &pinned) {
          auto *in_states = net->addInput("states", backbone_dt, nvinfer1::Dims2{-1, hidden});
          auto *in_last = net->addInput("last", backbone_dt, nvinfer1::Dims2{1, hidden});
          auto *s_f16 = ensure_dtype(net, in_states, nvinfer1::DataType::kHALF);
          auto *l_f16 = ensure_dtype(net, in_last, nvinfer1::DataType::kHALF);

          auto *profile = builder->createOptimizationProfile();
          profile->setDimensions("states", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, hidden});
          profile->setDimensions("states", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{std::min(64, max_seq), hidden});
          profile->setDimensions("states", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{max_seq, hidden});
          cfg->addOptimizationProfile(profile);

          for (const char *sname : {"Target", "Name", "Checkpoint", "DueAt"}) {
            for (const char *part : {"positions", "boundaries"}) {
              std::string prefix = std::string("spans.") + sname + "." + part;
              const auto *wt = gguf.tensor_info(prefix + ".weight");
              require(static_cast<int>(wt->ne[0]) == hidden * 2, "SPAN_WEIGHT_DIM_MISMATCH: " + prefix);
              int out_dim = static_cast<int>(wt->ne[1]);
              auto full_w = gguf.read_f16(prefix + ".weight");

              std::vector<uint16_t> w_left(static_cast<size_t>(out_dim) * hidden);
              std::vector<uint16_t> w_right(static_cast<size_t>(out_dim) * hidden);
              for (int o = 0; o < out_dim; ++o) {
                const uint16_t *row = full_w.data() + static_cast<size_t>(o) * (hidden * 2);
                std::memcpy(w_left.data() + static_cast<size_t>(o) * hidden, row, static_cast<size_t>(hidden) * sizeof(uint16_t));
                std::memcpy(w_right.data() + static_cast<size_t>(o) * hidden, row + hidden, static_cast<size_t>(hidden) * sizeof(uint16_t));
              }
              pinned.f16.push_back(std::move(w_left));
              const auto &wl_buf = pinned.f16.back();
              pinned.f16.push_back(std::move(w_right));
              const auto &wr_buf = pinned.f16.back();
              pinned.f16.push_back(gguf.read_f16(prefix + ".bias"));
              const auto &b_buf = pinned.f16.back();

              auto *wl_c = net->addConstant(nvinfer1::Dims2{out_dim, hidden},
                  nvinfer1::Weights{nvinfer1::DataType::kHALF, wl_buf.data(), static_cast<int64_t>(wl_buf.size())});
              auto *wr_c = net->addConstant(nvinfer1::Dims2{out_dim, hidden},
                  nvinfer1::Weights{nvinfer1::DataType::kHALF, wr_buf.data(), static_cast<int64_t>(wr_buf.size())});
              auto *b_c = net->addConstant(nvinfer1::Dims2{1, out_dim},
                  nvinfer1::Weights{nvinfer1::DataType::kHALF, b_buf.data(), static_cast<int64_t>(b_buf.size())});

              auto *mm_left = net->addMatrixMultiply(*s_f16, nvinfer1::MatrixOperation::kNONE,
                                                     *wl_c->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
              auto *mm_right = net->addMatrixMultiply(*l_f16, nvinfer1::MatrixOperation::kNONE,
                                                      *wr_c->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
              auto *right_b = net->addElementWise(*mm_right->getOutput(0), *b_c->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
              auto *sum = net->addElementWise(*mm_left->getOutput(0), *right_b->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
              auto *out = ensure_dtype(net, sum->getOutput(0), nvinfer1::DataType::kFLOAT);
              std::string out_name = std::string(sname) + "." + part;
              out->setName(out_name.c_str());
              net->markOutput(*out);
            }
          }
        });

    // 4 & 5. Generator engines (Task, Note): MatMul(1024->vocab) + SoftMax + TopK(k=1)
    auto build_gen = [&](const char *head_name, TrtSubEngine &eng) {
      load_or_build_subengine(
          eng,
          gguf_path + ".gen_" + head_name + "." + dt_tag + ".engine",
          gguf_path,
          [&](nvinfer1::IBuilder *, nvinfer1::INetworkDefinition *net, nvinfer1::IBuilderConfig *, PinnedBuffers &pinned) {
            auto *in = net->addInput("last", backbone_dt, nvinfer1::Dims2{1, hidden});
            auto *x = ensure_dtype(net, in, nvinfer1::DataType::kHALF);
            auto *logits_f16 = add_linear_f16(net, x, gguf, std::string("generators.") + head_name, false, pinned);
            auto *logits_f32 = ensure_dtype(net, logits_f16, nvinfer1::DataType::kFLOAT);
            auto *sm = net->addSoftMax(*logits_f32);
            require(sm != nullptr, "TRT_GEN_SOFTMAX_FAILED");
            sm->setAxes(1U << 1);
            auto *topk = net->addTopK(*sm->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, 1U << 1);
            require(topk != nullptr, "TRT_GEN_TOPK_FAILED");
            topk->getOutput(0)->setName("prob");
            topk->getOutput(1)->setName("token");
            net->markOutput(*topk->getOutput(0));
            net->markOutput(*topk->getOutput(1));
          });
    };
    build_gen("Task", gen_task_engine);
    build_gen("Note", gen_note_engine);
  }

  void run_classifiers(const void *d_pooled,
                       std::vector<float> &intent_logits,
                       std::vector<float> &op_logits,
                       std::vector<float> &prio_logits) {
    classifiers_engine.run(
        stream,
        {{"pooled", {d_pooled, nvinfer1::Dims2{1, hidden}}}},
        {{"Intention", &intent_logits}, {"Operation", &op_logits}, {"Priority", &prio_logits}});
  }

  void run_spans(const void *d_states, const void *d_last, int T,
                 std::unordered_map<std::string, std::vector<float>> &out) {
    std::vector<std::pair<std::string, std::vector<float> *>> float_outs;
    for (const char *sname : {"Target", "Name", "Checkpoint", "DueAt"}) {
      for (const char *part : {"positions", "boundaries"}) {
        std::string key = std::string(sname) + "." + part;
        float_outs.push_back({key, &out[key]});
      }
    }
    spans_engine.run(
        stream,
        {{"states", {d_states, nvinfer1::Dims2{T, hidden}}},
         {"last", {d_last, nvinfer1::Dims2{1, hidden}}}},
        float_outs);
  }

  std::pair<int32_t, float> run_generator_step(const std::string &head_name, const void *d_last) {
    TrtSubEngine &eng = (head_name == "Task") ? gen_task_engine : gen_note_engine;
    std::vector<float> prob_vec;
    int32_t tok = 0;
    eng.run(stream,
            {{"last", {d_last, nvinfer1::Dims2{1, hidden}}}},
            {{"prob", &prob_vec}},
            {{"token", &tok}});
    return {tok, prob_vec.empty() ? 0.0f : prob_vec[0]};
  }

  float run_confidence(const std::vector<float> &features) {
    require(features.size() == 9, "CONFIDENCE_FEATURE_DIM_MISMATCH");
    std::vector<float> score_vec;
    confidence_engine.run(
        stream,
        {},
        {{"score", &score_vec}},
        {},
        {{"features", {features.data(), nvinfer1::Dims2{1, 9}}}});
    require(score_vec.size() == 1, "CONFIDENCE_OUTPUT_DIM_MISMATCH");
    return score_vec[0];
  }
};

#endif

class TrtLanguageModel final : public Model {
  bool router = false;
  int hidden = 1024;
  int limit = 128;
  int n_ctx = 2048;
  OfficialTokenizer tokenizer;
  json metadata;
#if defined(DESKPET_HAS_TENSORRT)
  std::unique_ptr<TensorRtEngineRunner> trt_runner;
  std::unique_ptr<TrtHeadEngines> heads;
#endif

public:
  explicit TrtLanguageModel(const json &c) : Model(c), router(c.at("role").get<std::string>() == "router") {
    require(c.at("backend").get<std::string>() == "tensorrt", "BACKEND_UNSUPPORTED");
    auto device = c.at("device").get<std::string>();
    require(device == "cuda", "GPU_REQUIRED");

#if defined(DESKPET_HAS_TENSORRT)
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0, "GPU_BACKEND_UNAVAILABLE");
#else
    require(false, "TENSORRT_NOT_AVAILABLE");
#endif

    n_ctx = std::max(512, c.value("contextTokens", router ? 2560 : 2048));
    limit = std::max(1, c.value("maxTokens", 128));

    std::string primary_path = opt_path(c, "model");
    if (primary_path.empty()) primary_path = opt_path(c, "backbone");
    require(!primary_path.empty(), "MODEL_PATH_MISSING");
    std::string model_dir = dirname_of(primary_path);
    std::string models_root = dirname_of(model_dir);

    std::string tok_path = opt_path(c, "tokenizer");
    if (tok_path.empty()) {
      if (file_exists(model_dir + "/tokenizer/tokenizer.json")) tok_path = model_dir + "/tokenizer/tokenizer.json";
      else if (file_exists(model_dir + "/tokenizer.json")) tok_path = model_dir + "/tokenizer.json";
      else if (file_exists(model_dir + "/reference/tokenizer.json")) tok_path = model_dir + "/reference/tokenizer.json";
      else if (file_exists(models_root + "/router/tokenizer/tokenizer.json")) tok_path = models_root + "/router/tokenizer/tokenizer.json";
      else if (file_exists(models_root + "/router/reference/tokenizer.json")) tok_path = models_root + "/router/reference/tokenizer.json";
    }
    require(!tok_path.empty(), "TOKENIZER_NOT_FOUND");
    tokenizer.load(tok_path);

    std::string engine_path = opt_path(c, "engine");
    std::string onnx_path = opt_path(c, "onnx");
    auto ends_with = [](const std::string &s, const std::string &sfx) {
      return s.size() >= sfx.size() && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
    };
    if (engine_path.empty() && ends_with(primary_path, ".engine")) engine_path = primary_path;
    if (onnx_path.empty() && ends_with(primary_path, ".onnx")) onnx_path = primary_path;

    if (engine_path.empty()) {
      if (router) {
        if (file_exists(model_dir + "/backbone.engine")) engine_path = model_dir + "/backbone.engine";
        else if (file_exists(model_dir + "/model.engine")) engine_path = model_dir + "/model.engine";
        else engine_path = model_dir + "/backbone.engine";
      } else {
        engine_path = model_dir + "/model.engine";
      }
    }
    if (onnx_path.empty()) {
      if (router) {
        if (file_exists(model_dir + "/backbone.onnx")) onnx_path = model_dir + "/backbone.onnx";
        else onnx_path = model_dir + "/model.onnx";
      } else {
        onnx_path = model_dir + "/model.onnx";
      }
    }

#if defined(DESKPET_HAS_TENSORRT)
    trt_runner = std::make_unique<TensorRtEngineRunner>();
    trt_runner->load_or_build(engine_path, onnx_path, n_ctx, router);
    if (router && trt_runner->output_dim > 0) hidden = trt_runner->output_dim;

    if (router) {
      metadata = read_json(path("config"));
      heads = std::make_unique<TrtHeadEngines>();
      heads->init(path("heads"), trt_runner->runtime.get(), trt_runner->stream,
                  trt_runner->primary_output_dtype, hidden, n_ctx);
    }
#endif
  }

  ~TrtLanguageModel() override {
    stop();
#if defined(DESKPET_HAS_TENSORRT)
    heads.reset();
    trt_runner.reset();
#endif
  }

  std::vector<int32_t> tokenize_checked(const std::string &text) const {
    auto ids = tokenizer.encode(text);
    require(!ids.empty() && static_cast<int>(ids.size()) <= n_ctx - limit, "INPUT_TOKEN_LIMIT");
    return ids;
  }

  std::pair<std::string, double> generate_head(const std::string &head_name, const std::vector<int32_t> &prefix,
                                               const void *initial_last_gpu, Job &job, bool &finished) {
    finished = false;
    std::vector<int32_t> seq = prefix;
    std::vector<int32_t> generated_ids;
    double logp = 0.0;
    int count = 0;

#if defined(DESKPET_HAS_TENSORRT)
    const void *d_last = initial_last_gpu;
    const size_t elem_sz = dtype_bytes(trt_runner->primary_output_dtype);
    for (int i = 0; i < limit; ++i) {
      require(!job.cancelled.load(), "CANCELLED");
      if (i > 0) {
        if (static_cast<int>(seq.size()) >= n_ctx) break;
        auto view = trt_runner->execute_sequence_gpu(seq);
        require(view.feature_dim == hidden, "TRT_HIDDEN_DIM_MISMATCH");
        d_last = static_cast<const char *>(view.device_ptr) +
                 static_cast<size_t>(view.seq_dim - 1) * static_cast<size_t>(hidden) * elem_sz;
      }
      auto [token, p] = heads->run_generator_step(head_name, d_last);
      logp += std::log(std::max(static_cast<double>(p), 1e-30));
      ++count;
      if (tokenizer.is_eog(token)) {
        finished = true;
        break;
      }
      generated_ids.push_back(token);
      seq.push_back(token);
    }
#endif
    return {tokenizer.decode(generated_ids), count ? std::exp(logp / count) : 0.0};
  }

  json run_router(Job &job) {
    auto messages = job.input.at("messages");
    auto contextMode = job.input.value("contextMode", std::string("legacy"));
    require(contextMode == "legacy" || contextMode == "reference" || contextMode == "disabled",
            "ROUTER_CONTEXT_MODE_UNSUPPORTED");
    require(!messages.empty() && messages.back().at("role") == "user", "ROUTER_CURRENT_MESSAGE_REQUIRED");
    auto current = job.input.value("currentMessageIndex", static_cast<int>(messages.size()) - 1);
    require(current == static_cast<int>(messages.size()) - 1, "ROUTER_CURRENT_MESSAGE_INDEX_INVALID");
    if (contextMode == "disabled") {
      require(messages.size() == 1, "ROUTER_CONTEXT_DISABLED_REQUIRES_CURRENT_ONLY");
    }

    std::string prompt = render_messages(messages, true);
    auto ids = tokenize_checked(prompt);
    require(ids.size() <= metadata.at("max_input_tokens").get<size_t>(), "ROUTER_INPUT_TOKEN_LIMIT");
    require(!job.cancelled.load(), "CANCELLED");

#if defined(DESKPET_HAS_TENSORRT)
    const int T = static_cast<int>(ids.size());
    auto view = trt_runner->execute_sequence_gpu(ids);
    require(view.feature_dim == hidden && view.seq_dim == T, "TRT_HIDDEN_DIM_MISMATCH");
    const size_t elem_sz = dtype_bytes(view.dtype);
    const void *d_states = view.device_ptr;
    const void *d_pooled = static_cast<const char *>(d_states) +
                           static_cast<size_t>(T - 1) * static_cast<size_t>(hidden) * elem_sz;

    json values = json::object(), scores = json::object(), spans = json::object(), finished = json::object();
    std::vector<float> intent_logits, op_logits, prio_logits;
    heads->run_classifiers(d_pooled, intent_logits, op_logits, prio_logits);

    auto record_cls = [&](const std::string &name, const std::vector<float> &logits) {
      auto [index, score] = best_logit(logits);
      values[name] = metadata.at("labels").at(name).at(index);
      scores[name] = score;
    };
    record_cls("Intention", intent_logits);
    record_cls("Operation", op_logits);
    record_cls("Priority", prio_logits);

    // Marker-based message range alignment matching router.cpp
    struct Range {
      size_t a, b, left;
      int message;
    };
    std::vector<Range> ranges;
    auto marked = messages;
    std::vector<std::string> markers;
    for (size_t i = 0; i < messages.size(); ++i) {
      auto marker = "__deskpet_message_" + std::to_string(i) + "__";
      require(prompt.find(marker) == std::string::npos, "MESSAGE_MARKER_COLLISION");
      markers.push_back(marker);
      marked[i]["content"] = marker;
    }
    auto pattern = render_messages(marked, true);
    std::string mapped;
    size_t cursor = 0;
    for (size_t i = 0; i < messages.size(); ++i) {
      auto at = pattern.find(markers[i], cursor);
      require(at != std::string::npos, "MESSAGE_ALIGNMENT_FAILED");
      mapped += pattern.substr(cursor, at - cursor);
      std::string raw = messages[i].at("content").get<std::string>();
      auto [a, b] = trimmed_bounds(raw);
      size_t begin = mapped.size();
      mapped += raw.substr(a, b - a);
      ranges.push_back({begin, mapped.size(), unicode_scalars(raw.substr(0, a)), static_cast<int>(i)});
      cursor = at + markers[i].size();
    }
    mapped += pattern.substr(cursor);
    require(mapped == prompt, "MESSAGE_ALIGNMENT_FAILED");

    std::vector<std::pair<size_t, size_t>> offsets;
    offsets.reserve(ids.size());
    std::string restored;
    for (int32_t token : ids) {
      size_t a = restored.size();
      restored += tokenizer.raw_piece(token);
      offsets.emplace_back(a, restored.size());
    }
    require(restored == prompt, "TOKEN_OFFSET_ALIGNMENT_FAILED");
    for (auto &o : offsets) {
      while (o.first > 0 && (static_cast<uint8_t>(prompt[o.first]) & 0xC0) == 0x80) --o.first;
      while (o.second < prompt.size() && (static_cast<uint8_t>(prompt[o.second]) & 0xC0) == 0x80) ++o.second;
    }

    int chars = metadata.at("max_token_chars").get<int>() + 1;
    std::unordered_map<std::string, std::vector<float>> span_outs;
    if (values["Intention"] == "eureka") {
      heads->run_spans(d_states, d_pooled, T, span_outs);
    }

    for (const char *sname : {"Target", "Name", "Checkpoint", "DueAt"}) {
      values[sname] = nullptr;
      scores[sname] = 0.0;
      if (values["Intention"] != "eureka") continue;
      const auto &positions = span_outs.at(std::string(sname) + ".positions");
      const auto &boundaries = span_outs.at(std::string(sname) + ".boundaries");

      double max_score = -std::numeric_limits<double>::infinity();
      double sum = 0.0;
      int start = -1, end = -1, range_idx = -1;
      for (size_t r = 0; r < ranges.size(); ++r) {
        for (size_t a = 0; a < ids.size(); ++a) {
          if (offsets[a].first >= ranges[r].a && offsets[a].second <= ranges[r].b && offsets[a].first < offsets[a].second) {
            for (size_t b = a; b < ids.size() && offsets[b].second <= ranges[r].b; ++b) {
              double score = positions[2 * a] + positions[2 * b + 1];
              if (score > max_score) {
                sum = sum * std::exp(max_score - score) + 1.0;
                max_score = score;
                start = static_cast<int>(a);
                end = static_cast<int>(b);
                range_idx = static_cast<int>(r);
              } else {
                sum += std::exp(score - max_score);
              }
            }
          }
        }
      }
      if (start < 0) continue;

      std::string s = prompt.substr(offsets[start].first, offsets[start].second - offsets[start].first);
      std::string e = prompt.substr(offsets[end].first, offsets[end].second - offsets[end].first);
      size_t sb_len = std::min(static_cast<size_t>(chars), unicode_scalars(s));
      size_t eb_len = std::min(static_cast<size_t>(chars), unicode_scalars(e) + 1);
      auto sb = std::vector<float>(boundaries.begin() + start * chars * 2,
                                   boundaries.begin() + start * chars * 2 + sb_len);
      auto eb = std::vector<float>(boundaries.begin() + end * chars * 2 + chars,
                                   boundaries.begin() + end * chars * 2 + chars + eb_len);
      if (sb.empty() || eb.size() < 2) continue;
      eb[0] = -std::numeric_limits<float>::infinity();

      auto [si, sp] = best_logit(sb);
      auto [ei, ep] = best_logit(eb);
      size_t a = offsets[start].first + scalar_to_byte(s, static_cast<size_t>(si));
      size_t b = offsets[end].first + scalar_to_byte(e, static_cast<size_t>(ei));
      if (a >= b) {
        double best_pair = -std::numeric_limits<double>::infinity();
        int bs = -1, be = -1;
        for (size_t i = 0; i < sb.size(); ++i) {
          for (size_t j = i + 1; j < eb.size(); ++j) {
            if (static_cast<double>(sb[i]) + eb[j] > best_pair) {
              best_pair = static_cast<double>(sb[i]) + eb[j];
              bs = static_cast<int>(i);
              be = static_cast<int>(j);
            }
          }
        }
        if (bs < 0) continue;
        auto prob_of = [](const std::vector<float> &v, int idx) {
          float m = *std::max_element(v.begin(), v.end());
          double ssum = 0.0;
          for (float f : v) ssum += std::exp(static_cast<double>(f - m));
          return std::exp(static_cast<double>(v[idx] - m)) / ssum;
        };
        si = bs;
        ei = be;
        sp = prob_of(sb, si);
        ep = prob_of(eb, ei);
        a = offsets[start].first + scalar_to_byte(s, static_cast<size_t>(si));
        b = offsets[end].first + scalar_to_byte(e, static_cast<size_t>(ei));
      }
      if (a >= b) continue;
      const auto &r = ranges[range_idx];
      std::string text = prompt.substr(a, b - a);
      spans[sname] = {
          {"message", r.message},
          {"start", r.left + unicode_scalars(prompt.substr(r.a, a - r.a))},
          {"end", r.left + unicode_scalars(prompt.substr(r.a, b - r.a))},
          {"text", text},
      };
      values[sname] = text;
      scores[sname] = std::cbrt(std::max(1e-12, sp * ep / sum));
    }

    auto operation = values["Operation"].get<std::string>();
    bool isEureka = values["Intention"] == "eureka";
    bool do_task = isEureka && (operation == "create" || operation == "add_task");
    bool do_note = isEureka && (operation == "create" || operation == "add_note" || operation == "update_item");
    for (const auto &name : std::vector<std::string>{"Task", "Note"}) {
      values[name] = nullptr;
      scores[name] = 0.0;
      if ((name == "Task" && do_task) || (name == "Note" && do_note)) {
        const void *cur_pooled = d_pooled;
        if (name == "Note" && do_task) {
          auto rview = trt_runner->execute_sequence_gpu(ids);
          cur_pooled = static_cast<const char *>(rview.device_ptr) +
                       static_cast<size_t>(rview.seq_dim - 1) * static_cast<size_t>(hidden) * elem_sz;
        }
        bool done = false;
        auto value = generate_head(name, ids, cur_pooled, job, done);
        values[name] = value.first;
        scores[name] = value.second;
        finished[name] = done;
        require(done, "ROUTER_GENERATION_TRUNCATED");
      }
    }

    std::vector<float> features;
    features.reserve(9);
    for (const auto &h : std::vector<std::string>{"Intention", "Operation", "Priority", "Target", "Name", "Checkpoint", "DueAt", "Task", "Note"}) {
      bool active = (h == "Intention") || (isEureka && (h != "Task" || do_task) && (h != "Note" || do_note));
      features.push_back(active ? scores[h].get<float>() : 0.0f);
    }
    require(metadata.value("confidence_calibrated", false), "ROUTER_CONFIDENCE_UNAVAILABLE");
    float conf_score = heads->run_confidence(features);

    auto head_obj = [&](const std::string &h) {
      return json{{"value", isEureka ? values[h] : json(nullptr)},
                  {"confidence", isEureka ? scores[h] : json(0.0)}};
    };
    std::string intent = "unclear";
    if (values["Intention"] == "talk") intent = "chat";
    else if (values["Intention"] == "llm") intent = "llm";
    else if (isEureka) intent = "eureka." + operation;

    return {
        {"intention", {{"value", intent}, {"confidence", scores["Intention"]}}},
        {"modelIntention", {{"value", values["Intention"]}, {"confidence", scores["Intention"]}}},
        {"operation", head_obj("Operation")},
        {"target", head_obj("Target")},
        {"name", head_obj("Name")},
        {"checkpoint", head_obj("Checkpoint")},
        {"dueAt", {{"value", nullptr}, {"confidence", scores["DueAt"]}}},
        {"priority", head_obj("Priority")},
        {"task", head_obj("Task")},
        {"note", head_obj("Note")},
        {"spans", spans},
        {"modelConfidence", conf_score},
        {"generationFinished", finished},
    };
#else
    require(false, "TENSORRT_NOT_AVAILABLE");
    return json::object();
#endif
  }

  json run_output(Job &job) {
    auto messages = job.input.at("messages");
    std::string prompt = render_messages(messages, false);
    auto tokens = tokenize_checked(prompt);
    std::vector<int32_t> generated_ids;

#if defined(DESKPET_HAS_TENSORRT)
    for (int i = 0; i < limit; ++i) {
      require(!job.cancelled.load(), "CANCELLED");
      int32_t next = trt_runner->forward_next_token_gpu(tokens);
      if (tokenizer.is_eog(next)) break;
      generated_ids.push_back(next);
      tokens.push_back(next);
      if (static_cast<int>(tokens.size()) >= n_ctx) break;
    }
#else
    require(false, "TENSORRT_NOT_AVAILABLE");
#endif

    std::string text = tokenizer.decode(generated_ids);
    require(text.find("<think>") == std::string::npos, "UNEXPECTED_REASONING_OUTPUT");
    return {{"text", text}};
  }

  json infer(Job &job) override {
    return router ? run_router(job) : run_output(job);
  }
};

} // namespace

std::shared_ptr<Model> make_language_model(const json &config) {
  return std::make_shared<TrtLanguageModel>(config);
}
