#include "runtime.hpp"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_map>

#if defined(DESKPET_HAS_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace deskpet {
namespace {

constexpr const char *kIntentions[] = {"unclear", "talk", "github", "eureka", "llm", "answer"};
constexpr const char *kOperations[] = {
  "none", "query_tasks", "query_task_detail", "query_progress", "query_overdue", "query_due_soon",
  "query_Work_By_Milestone", "query_Milestones", "query_actors", "complete_task", "complete_Milestone",
  "start_task", "block_task", "postpone_task", "update_task_fields", "update_Milestone_Status", "create_task",
  "add_Milestone", "delete_task", "delete_Milestone", "add_task_note", "pr_list", "pr_view", "pr_diff",
  "pr_Approve", "pr_comment", "issue_list", "issue_view", "commit_summary", "yes", "no", "cancel",
  "confirm_Other"
};
constexpr const char *kPriorities[] = {"low", "medium", "high", "urgent"};
constexpr const char *kPositionKinds[] = {"none", "single", "range"};
constexpr const char *kSpans[] = {"Target", "Name", "Checkpoint", "DueAt"};
constexpr int kAllowed[6][33] = {
  {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0},
  {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
  {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
};

bool file_exists(const std::string &path) {
  if (path.empty()) return false;
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

std::string dirname_of(const std::string &path) {
  auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0, pos);
}

size_t unicode_scalars(const std::string &s, size_t bytes) {
  size_t count = 0;
  for (size_t i = 0; i < bytes && i < s.size(); ++i)
    if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) ++count;
  return count;
}

size_t trim_left(const std::string &s, size_t b, size_t e) {
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  return b;
}

size_t trim_right(const std::string &s, size_t b, size_t e) {
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return e;
}

std::vector<float> softmax(const float *x, size_t n) {
  float m = *std::max_element(x, x + n), sum = 0;
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) sum += (out[i] = std::exp(x[i] - m));
  for (float &v : out) v /= sum;
  return out;
}

size_t argmax(const std::vector<float> &p) {
  return static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
}

std::string map_route(const std::string &intent, const std::string &op) {
  if (intent == "talk") return "chat";
  if (intent == "unclear") return "clarify";
  if (intent == "llm") return "llm";
  if (intent == "answer") {
    if (op == "yes") return "confirm.yes";
    if (op == "no") return "confirm.no";
    if (op == "cancel") return "confirm.cancel";
    return "confirm.other";
  }
  if (intent == "github") {
    if (op == "pr_list") return "pr.list";
    if (op == "pr_view") return "pr.view";
    if (op == "pr_diff") return "pr.diff";
    if (op == "pr_Approve") return "pr.approve";
    if (op == "pr_comment") return "pr.comment";
    if (op == "issue_list") return "issue.list";
    if (op == "issue_view") return "issue.view";
    if (op == "commit_summary") return "commit.summary";
    return "llm";
  }
  if (intent == "eureka" && op != "none") return "eureka." + op;
  return "llm";
}

struct HfTokenizer {
  std::unordered_map<std::string, int32_t> token_to_id;
  std::unordered_map<int32_t, std::string> id_to_token;
  std::vector<std::pair<std::string, int32_t>> special_tokens;
  std::string byte_to_unicode[256];
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
  int32_t eos_id = 151645;
  int32_t eot_id = 151643;

  static std::string codepoint_to_utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
  }

  void init_byte_tables() {
    bool in_bs[256] = {};
    auto mark = [&](int a, int b) { for (int i = a; i <= b; ++i) in_bs[i] = true; };
    mark(33, 126); mark(161, 172); mark(174, 255);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp = in_bs[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
      byte_to_unicode[b] = codepoint_to_utf8(cp);
      unicode_to_byte[cp] = static_cast<uint8_t>(b);
    }
  }

  void load(const std::string &path) {
    init_byte_tables();
    std::ifstream f(path);
    if (!f) fail("TOKENIZER_NOT_FOUND", path);
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded()) fail("TOKENIZER_INVALID_JSON", path);
    if (j.contains("model") && j["model"].contains("vocab") && j["model"]["vocab"].is_object()) {
      for (auto it = j["model"]["vocab"].begin(); it != j["model"]["vocab"].end(); ++it) {
        int32_t id = it.value().get<int32_t>();
        token_to_id[it.key()] = id;
        id_to_token[id] = it.key();
      }
    }
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
      for (const auto &at : j["added_tokens"]) {
        if (at.contains("content") && at.contains("id")) {
          std::string c = at["content"].get<std::string>();
          int32_t id = at["id"].get<int32_t>();
          token_to_id[c] = id;
          id_to_token[id] = c;
          special_tokens.push_back({c, id});
          if (c == "<|im_end|>") eos_id = id;
          if (c == "<|endoftext|>") eot_id = id;
        }
      }
    }
    std::sort(special_tokens.begin(), special_tokens.end(), [](const auto &a, const auto &b) {
      return a.first.size() > b.first.size();
    });
  }

  std::vector<int32_t> encode_chunk(const std::string &raw) const {
    std::string encoded;
    for (unsigned char c : raw) encoded += byte_to_unicode[c];
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < encoded.size()) {
      size_t best_len = 0;
      int32_t best_id = -1;
      size_t max_l = std::min<size_t>(encoded.size() - i, 64);
      for (size_t l = max_l; l >= 1; --l) {
        auto it = token_to_id.find(encoded.substr(i, l));
        if (it != token_to_id.end()) {
          best_len = l;
          best_id = it->second;
          break;
        }
      }
      if (best_id >= 0) {
        ids.push_back(best_id);
        i += best_len;
      } else {
        ++i;
      }
    }
    return ids;
  }

  std::vector<int32_t> encode(const std::string &text) const {
    std::vector<int32_t> out;
    size_t pos = 0;
    while (pos < text.size()) {
      size_t best_pos = std::string::npos;
      size_t best_len = 0;
      int32_t best_id = -1;
      for (const auto &sp : special_tokens) {
        size_t found = text.find(sp.first, pos);
        if (found != std::string::npos && (best_pos == std::string::npos || found < best_pos)) {
          best_pos = found;
          best_len = sp.first.size();
          best_id = sp.second;
        }
      }
      if (best_pos == std::string::npos) {
        auto sub = encode_chunk(text.substr(pos));
        out.insert(out.end(), sub.begin(), sub.end());
        break;
      }
      if (best_pos > pos) {
        auto sub = encode_chunk(text.substr(pos, best_pos - pos));
        out.insert(out.end(), sub.begin(), sub.end());
      }
      out.push_back(best_id);
      pos = best_pos + best_len;
    }
    return out;
  }

  std::string piece_of(int32_t id) const {
    auto it = id_to_token.find(id);
    if (it == id_to_token.end()) return {};
    const std::string &tok = it->second;
    for (const auto &sp : special_tokens) {
      if (sp.second == id) return sp.first;
    }
    std::string bytes;
    size_t i = 0;
    while (i < tok.size()) {
      unsigned char c = static_cast<unsigned char>(tok[i]);
      uint32_t cp = 0;
      size_t step = 1;
      if ((c & 0x80) == 0) { cp = c; step = 1; }
      else if ((c & 0xE0) == 0xC0 && i + 1 < tok.size()) {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(tok[i + 1]) & 0x3F);
        step = 2;
      } else if ((c & 0xF0) == 0xE0 && i + 2 < tok.size()) {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(tok[i + 1]) & 0x3F) << 6) | (static_cast<unsigned char>(tok[i + 2]) & 0x3F);
        step = 3;
      }
      auto bit = unicode_to_byte.find(cp);
      if (bit != unicode_to_byte.end()) bytes.push_back(static_cast<char>(bit->second));
      i += step;
    }
    return bytes;
  }
};

std::string render_chatml(const std::vector<std::pair<std::string, std::string>> &msgs, bool add_generation_prompt) {
  std::string out;
  for (const auto &m : msgs) {
    out += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
  }
  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  }
  return out;
}

#if defined(DESKPET_HAS_TENSORRT)
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
  if (!f) fail("TRT_ENGINE_READ_FAILED", path);
  auto sz = f.tellg();
  if (sz <= 0) fail("TRT_ENGINE_EMPTY", path);
  std::vector<char> buf(static_cast<size_t>(sz));
  f.seekg(0, std::ios::beg);
  f.read(buf.data(), sz);
  return buf;
}

void write_binary_file(const std::string &path, const void *data, size_t size) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (f) f.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
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

struct TensorRtEngineRunner {
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  std::string input_name;
  std::string output_name;
  nvinfer1::DataType input_dtype = nvinfer1::DataType::kINT32;
  nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;
  int hidden_or_vocab_dim = 0;

  ~TensorRtEngineRunner() {
    if (stream) cudaStreamDestroy(stream);
  }

  void load_or_build(const std::string &engine_path, const std::string &onnx_path, int max_seq) {
    if (cudaStreamCreate(&stream) != cudaSuccess) fail("CUDA_STREAM_FAILED", "cudaStreamCreate failed");
    runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
    if (!runtime) fail("TRT_RUNTIME_FAILED", "createInferRuntime failed");

    if (file_exists(engine_path)) {
      auto bytes = read_binary_file(engine_path);
      engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    } else if (file_exists(onnx_path)) {
      TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_trt_logger));
      TrtUniquePtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(
          1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
      TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_trt_logger));
      if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fail("TRT_ONNX_PARSE_FAILED", onnx_path);
      }
      TrtUniquePtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
      // Enforce INT4 + FP16 and cap workspace memory to 512MB for Jetson Orin Nano unified memory safety
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
      if (!plan) fail("TRT_BUILD_FAILED", onnx_path);
      if (!engine_path.empty()) write_binary_file(engine_path, plan->data(), plan->size());
      engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    } else {
      fail("TRT_MODEL_MISSING", "TensorRT .engine or .onnx file not found: " + engine_path);
    }

    if (!engine) fail("TRT_ENGINE_INIT_FAILED", engine_path);
    context.reset(engine->createExecutionContext());
    if (!context) fail("TRT_CONTEXT_INIT_FAILED", engine_path);

    int nb_io = engine->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
      const char *name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT && input_name.empty()) {
        input_name = name;
        input_dtype = engine->getTensorDataType(name);
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name.empty()) {
        output_name = name;
        output_dtype = engine->getTensorDataType(name);
        auto dims = engine->getTensorShape(name);
        if (dims.nbDims >= 1) hidden_or_vocab_dim = static_cast<int>(dims.d[dims.nbDims - 1]);
      }
    }
    if (input_name.empty() || output_name.empty()) fail("TRT_IO_TENSOR_MISSING", engine_path);
  }

  std::vector<float> forward_sequence(const std::vector<int32_t> &tokens) {
    int seq_len = static_cast<int>(tokens.size());
    if (hidden_or_vocab_dim <= 0) fail("TRT_OUTPUT_DIM_INVALID", output_name);

    int nb_io = engine->getNbIOTensors();
    std::vector<void *> dev_buffers;
    void *d_out = nullptr;
    size_t out_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(hidden_or_vocab_dim);
    size_t out_bytes = out_elems * dtype_bytes(output_dtype);

    for (int i = 0; i < nb_io; ++i) {
      const char *name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      auto dt = engine->getTensorDataType(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        context->setInputShape(name, nvinfer1::Dims2{1, seq_len});
        void *d_in = nullptr;
        if (dt == nvinfer1::DataType::kINT64) {
          std::vector<int64_t> ids64(tokens.begin(), tokens.end());
          if (std::string(name).find("mask") != std::string::npos) std::fill(ids64.begin(), ids64.end(), 1);
          cudaMalloc(&d_in, ids64.size() * sizeof(int64_t));
          cudaMemcpyAsync(d_in, ids64.data(), ids64.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        } else {
          std::vector<int32_t> ids32 = tokens;
          if (std::string(name).find("mask") != std::string::npos) std::fill(ids32.begin(), ids32.end(), 1);
          cudaMalloc(&d_in, ids32.size() * sizeof(int32_t));
          cudaMemcpyAsync(d_in, ids32.data(), ids32.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        }
        dev_buffers.push_back(d_in);
        context->setTensorAddress(name, d_in);
      } else if (std::string(name) == output_name) {
        cudaMalloc(&d_out, out_bytes);
        dev_buffers.push_back(d_out);
        context->setTensorAddress(name, d_out);
      }
    }

    if (!context->enqueueV3(stream)) {
      for (void *p : dev_buffers) cudaFree(p);
      fail("TRT_ENQUEUE_FAILED", "TensorRT enqueueV3 failed");
    }

    std::vector<float> host_out(out_elems);
    if (output_dtype == nvinfer1::DataType::kHALF) {
      std::vector<uint16_t> fp16_out(out_elems);
      cudaMemcpyAsync(fp16_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      for (size_t i = 0; i < out_elems; ++i) {
        __half h;
        std::memcpy(&h, &fp16_out[i], sizeof(uint16_t));
        host_out[i] = __half2float(h);
      }
    } else {
      cudaMemcpyAsync(host_out.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }

    for (void *p : dev_buffers) cudaFree(p);
    return host_out;
  }
};
#endif

struct TrtLanguageModel final : Model {
  std::string role_name;
  int n_ctx = 2048, max_tokens = 64;
  HfTokenizer tokenizer;
  gguf_context *head_gguf = nullptr;
  ggml_context *head_mem = nullptr;
#if defined(DESKPET_HAS_TENSORRT)
  std::unique_ptr<TensorRtEngineRunner> trt_runner;
#endif

  explicit TrtLanguageModel(const json &c) {
    role_name = req_str(c, "role");
    auto device = req_str(c, "device");
    if (device != "cuda") fail("GPU_BACKEND_UNAVAILABLE", "TensorRT requires cuda device");

#if defined(DESKPET_HAS_TENSORRT)
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      fail("GPU_DEVICE_UNAVAILABLE", "CUDA/TensorRT device not found");
    }
#else
    fail("TENSORRT_NOT_AVAILABLE", "Built without TensorRT headers/libraries");
#endif

    n_ctx = std::max(512, c.value("contextTokens", role_name == "router" ? 2560 : 2048));
    max_tokens = std::max(1, c.value("maxTokens", role_name == "router" ? 48 : 96));

    std::string primary_path = opt_path_arg(c, role_name == "router" ? "backbone" : "model");
    std::string model_dir = dirname_of(primary_path);
    std::string models_root = dirname_of(model_dir);

    std::string tok_path = opt_path_arg(c, "tokenizer");
    if (tok_path.empty()) {
      if (file_exists(model_dir + "/tokenizer/tokenizer.json")) tok_path = model_dir + "/tokenizer/tokenizer.json";
      else if (file_exists(model_dir + "/reference/tokenizer.json")) tok_path = model_dir + "/reference/tokenizer.json";
      else if (file_exists(models_root + "/router/tokenizer/tokenizer.json")) tok_path = models_root + "/router/tokenizer/tokenizer.json";
      else if (file_exists(models_root + "/router/reference/tokenizer.json")) tok_path = models_root + "/router/reference/tokenizer.json";
    }
    tokenizer.load(tok_path);

    std::string engine_path = opt_path_arg(c, "engine");
    std::string onnx_path = opt_path_arg(c, "onnx");
    if (engine_path.empty()) {
      if (role_name == "router") {
        if (file_exists(model_dir + "/backbone.engine")) engine_path = model_dir + "/backbone.engine";
        else if (file_exists(model_dir + "/model.engine")) engine_path = model_dir + "/model.engine";
        else engine_path = model_dir + "/backbone.engine";
      } else {
        engine_path = model_dir + "/model.engine";
      }
    }
    if (onnx_path.empty()) {
      if (role_name == "router") {
        if (file_exists(model_dir + "/backbone.onnx")) onnx_path = model_dir + "/backbone.onnx";
        else onnx_path = model_dir + "/model.onnx";
      } else {
        onnx_path = model_dir + "/model.onnx";
      }
    }

#if defined(DESKPET_HAS_TENSORRT)
    trt_runner = std::make_unique<TensorRtEngineRunner>();
    trt_runner->load_or_build(engine_path, onnx_path, n_ctx);
#endif

    if (role_name == "router") {
      auto hpath = path_arg(c, "heads");
      gguf_init_params gp = {false, &head_mem};
      head_gguf = gguf_init_from_file(hpath.c_str(), gp);
      if (!head_gguf || !head_mem) fail("HEAD_LOAD_FAILED", hpath);
    }
  }

  ~TrtLanguageModel() override {
    stop_worker();
#if defined(DESKPET_HAS_TENSORRT)
    trt_runner.reset();
#endif
    if (head_gguf) gguf_free(head_gguf);
    if (head_mem) ggml_free(head_mem);
  }

  ggml_tensor *tensor(const std::string &name) const {
    auto *t = ggml_get_tensor(head_mem, name.c_str());
    if (!t) fail("HEAD_TENSOR_MISSING", name);
    return t;
  }

  static float read_elem(const ggml_tensor *t, size_t idx) {
    if (t->type == GGML_TYPE_F16) {
      ggml_fp16_t v;
      std::memcpy(&v, static_cast<const char *>(t->data) + idx * sizeof(ggml_fp16_t), sizeof(v));
      return ggml_fp16_to_fp32(v);
    }
    return reinterpret_cast<const float *>(t->data)[idx];
  }

  static void linear_f16(const ggml_tensor *w, const ggml_tensor *b, const float *in, size_t in_dim, float *out, size_t out_dim) {
    for (size_t o = 0; o < out_dim; ++o) {
      float sum = b ? read_elem(b, o) : 0.0f;
      size_t row = o * in_dim;
      for (size_t i = 0; i < in_dim; ++i) sum += read_elem(w, row + i) * in[i];
      out[o] = sum;
    }
  }

  std::pair<std::string, bool> generate_head(const std::string &head_name, const std::vector<int32_t> &prompt, const float *first_hidden, int cap, const std::atomic<bool> &cancelled) {
    auto prefix = tokenizer.encode("<|" + head_name + "|>");
    if (prefix.size() != 1) fail("SPECIAL_TOKEN_MISSING", head_name);
    std::vector<int32_t> seq = prompt;
    seq.push_back(prefix[0]);
    std::string out;
    std::vector<float> cur_hidden(first_hidden, first_hidden + 1024);
    auto *w = tensor("generators." + head_name + ".weight");
    size_t vocab_sz = static_cast<size_t>(w->ne[1]);
    size_t h_dim = static_cast<size_t>(w->ne[0]);

    for (int step = 0; step < cap; ++step) {
      if (cancelled.load()) fail("CANCELLED", "router generation cancelled");
      if (step > 0) {
        if (static_cast<int>(seq.size()) >= n_ctx) return {out, true};
#if defined(DESKPET_HAS_TENSORRT)
        auto all_h = trt_runner->forward_sequence(seq);
        std::memcpy(cur_hidden.data(), all_h.data() + (seq.size() - 1) * 1024, 1024 * sizeof(float));
#endif
      }
      int32_t next = 0;
      float best = -1e30f;
      for (size_t v = 0; v < vocab_sz; ++v) {
        float sum = 0.0f;
        size_t row = v * h_dim;
        for (size_t i = 0; i < h_dim; ++i) sum += read_elem(w, row + i) * cur_hidden[i];
        if (sum > best) { best = sum; next = static_cast<int32_t>(v); }
      }
      if (next == tokenizer.eos_id || next == tokenizer.eot_id) return {out, false};
      out += tokenizer.piece_of(next);
      seq.push_back(next);
    }
    return {out, true};
  }

  json run_router(const json &in, const std::atomic<bool> &cancelled) {
    std::string text = req_str(in, "text");
    std::vector<std::pair<std::string, std::string>> msgs;
    if (in.contains("messages") && in["messages"].is_array()) {
      for (const auto &m : in["messages"])
        if (m.contains("role") && m.contains("content"))
          msgs.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
    }
    std::string prefix_prompt = msgs.empty() ? std::string() : render_chatml(msgs, false);
    msgs.push_back({"user", text});
    std::string full_prompt = render_chatml(msgs, true);
    size_t text_pos = full_prompt.find(text, prefix_prompt.size());
    if (text_pos == std::string::npos) text_pos = full_prompt.rfind(text);
    if (text_pos == std::string::npos) fail("PROMPT_SPAN_MISMATCH", "utterance not found in rendered prompt");

    auto all_tokens = tokenizer.encode(full_prompt);
    auto tokens = all_tokens;
    if (static_cast<int>(tokens.size()) > n_ctx - 4)
      tokens.erase(tokens.begin(), tokens.end() - (n_ctx - 4));
    int dropped = static_cast<int>(all_tokens.size()) - static_cast<int>(tokens.size());

#if defined(DESKPET_HAS_TENSORRT)
    std::vector<float> emb = trt_runner->forward_sequence(tokens);
#else
    std::vector<float> emb;
#endif
    if (cancelled.load()) fail("CANCELLED", "router cancelled");

    int T = static_cast<int>(tokens.size());
    const float *pooled = emb.data() + static_cast<size_t>(T - 1) * 1024;

    float intent_logits[6] = {}, op_logits[33] = {}, prio_logits[4] = {};
    linear_f16(tensor("classifiers.Intention.weight"), tensor("classifiers.Intention.bias"), pooled, 1024, intent_logits, 6);
    linear_f16(tensor("classifiers.Operation.weight"), tensor("classifiers.Operation.bias"), pooled, 1024, op_logits, 33);
    linear_f16(tensor("classifiers.Priority.weight"), tensor("classifiers.Priority.bias"), pooled, 1024, prio_logits, 4);

    auto intent_p = softmax(intent_logits, 6);
    auto prio_p = softmax(prio_logits, 4);
    size_t intent_idx = argmax(intent_p), prio_idx = argmax(prio_p);
    float masked_op[33];
    for (int i = 0; i < 33; ++i) masked_op[i] = kAllowed[intent_idx][i] ? op_logits[i] : -1e9f;
    auto op_p = softmax(masked_op, 33);
    size_t op_idx = argmax(op_p);

    float conf_in[1067], conf_h[256] = {}, conf_logit[1] = {};
    std::memcpy(conf_in, pooled, 1024 * sizeof(float));
    std::memcpy(conf_in + 1024, intent_logits, 6 * sizeof(float));
    std::memcpy(conf_in + 1030, masked_op, 33 * sizeof(float));
    std::memcpy(conf_in + 1063, prio_logits, 4 * sizeof(float));
    linear_f16(tensor("confidence.0.weight"), tensor("confidence.0.bias"), conf_in, 1067, conf_h, 256);
    for (float &v : conf_h) v = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
    linear_f16(tensor("confidence.2.weight"), tensor("confidence.2.bias"), conf_h, 256, conf_logit, 1);
    float model_conf = 1.0f / (1.0f + std::exp(-conf_logit[0]));

    struct TokSpan { size_t b, e; };
    std::vector<TokSpan> tok_spans(T);
    size_t cursor = 0;
    for (int i = 0; i < dropped && i < static_cast<int>(all_tokens.size()); ++i)
      cursor += tokenizer.piece_of(all_tokens[i]).size();
    for (int i = 0; i < T; ++i) {
      auto p = tokenizer.piece_of(tokens[i]);
      tok_spans[i] = {cursor, cursor + p.size()};
      cursor += p.size();
    }
    size_t u_begin = text_pos, u_end = text_pos + text.size();

    json spans = json::object();
    json due_span = nullptr, target_span = nullptr, name_span = nullptr, checkpoint_span = nullptr;

    for (const char *sname : kSpans) {
      auto *pw = tensor(std::string("spans.") + sname + ".positions.weight");
      auto *pb = tensor(std::string("spans.") + sname + ".positions.bias");
      auto *bw = tensor(std::string("spans.") + sname + ".boundaries.weight");
      auto *bb = tensor(std::string("spans.") + sname + ".boundaries.bias");

      float pos_logits[3] = {};
      linear_f16(pw, pb, pooled, 1024, pos_logits, 3);
      auto pos_p = softmax(pos_logits, 3);
      size_t kind_idx = argmax(pos_p);
      std::string kind = kPositionKinds[kind_idx];

      std::vector<float> start_l(T, -1e9f), end_l(T, -1e9f);
      bool any_valid = false;
      for (int t = 0; t < T; ++t) {
        if (tok_spans[t].e > u_begin && tok_spans[t].b < u_end) {
          float feat[2048];
          std::memcpy(feat, emb.data() + static_cast<size_t>(t) * 1024, 1024 * sizeof(float));
          std::memcpy(feat + 1024, pooled, 1024 * sizeof(float));
          float b_out[2] = {};
          linear_f16(bw, bb, feat, 2048, b_out, 2);
          start_l[t] = b_out[0];
          end_l[t] = b_out[1];
          any_valid = true;
        }
      }
      auto sp = softmax(start_l.data(), T), ep = softmax(end_l.data(), T);
      size_t si = argmax(sp), ei = si;
      if (kind == "range") {
        float best = -1;
        for (size_t s = 0; s < static_cast<size_t>(T); ++s)
          for (size_t e = s; e < static_cast<size_t>(T); ++e)
            if (tok_spans[s].e > u_begin && tok_spans[e].b < u_end && sp[s] * ep[e] > best) {
              best = sp[s] * ep[e]; si = s; ei = e;
            }
      }
      if (kind == "none" || !any_valid) {
        spans[sname] = {{"kind", kind}, {"confidence", pos_p[kind_idx]}, {"startToken", si}, {"endToken", ei}, {"startScalar", nullptr}, {"endScalar", nullptr}, {"value", nullptr}};
        continue;
      }
      size_t b = std::max(tok_spans[si].b, u_begin), e = std::min(tok_spans[ei].e, u_end);
      while (b > u_begin && (static_cast<unsigned char>(full_prompt[b]) & 0xC0) == 0x80) --b;
      while (e < u_end && (static_cast<unsigned char>(full_prompt[e]) & 0xC0) == 0x80) ++e;
      b = trim_left(full_prompt, b, e);
      e = trim_right(full_prompt, b, e);
      double conf = kind == "single" ? pos_p[kind_idx] * sp[si] : pos_p[kind_idx] * std::sqrt(sp[si] * ep[ei]);
      if (e <= b) {
        spans[sname] = {{"kind", kind}, {"confidence", conf}, {"startToken", si}, {"endToken", ei}, {"startScalar", nullptr}, {"endScalar", nullptr}, {"value", nullptr}};
        continue;
      }
      size_t rel_b = b - u_begin, rel_e = e - u_begin;
      std::string val = text.substr(rel_b, rel_e - rel_b);
      json entry = {
        {"kind", kind}, {"confidence", conf}, {"startToken", si}, {"endToken", ei},
        {"startScalar", unicode_scalars(text, rel_b)}, {"endScalar", unicode_scalars(text, rel_e)}, {"value", val}
      };
      spans[sname] = entry;
      if (std::string(sname) == "DueAt") due_span = val;
      if (std::string(sname) == "Target") target_span = val;
      if (std::string(sname) == "Name") name_span = val;
      if (std::string(sname) == "Checkpoint") checkpoint_span = val;
    }

    int gen_cap = std::min(max_tokens, 48);
    auto [task_raw, task_trunc] = generate_head("Task", tokens, pooled, gen_cap, cancelled);
    auto [note_raw, note_trunc] = generate_head("Note", tokens, pooled, gen_cap, cancelled);

    auto clean = [](std::string s) -> json {
      size_t b = trim_left(s, 0, s.size()), e = trim_right(s, b, s.size());
      s = s.substr(b, e - b);
      if (s.empty() || s == "none" || s == "null") return nullptr;
      return s;
    };
    std::string m_intent = kIntentions[intent_idx], op_name = kOperations[op_idx];
    std::string mapped = map_route(m_intent, op_name);
    return {
      {"intention", mapped}, {"modelIntention", m_intent}, {"intentionScore", intent_p[intent_idx]},
      {"operation", op_name}, {"operationScore", op_p[op_idx]},
      {"priority", kPriorities[prio_idx]}, {"priorityScore", prio_p[prio_idx]},
      {"confidence", model_conf}, {"modelConfidence", model_conf},
      {"target", target_span}, {"name", name_span}, {"checkpoint", checkpoint_span},
      {"dueAt", due_span}, {"task", clean(task_raw)}, {"note", clean(note_raw)},
      {"taskTruncated", task_trunc}, {"noteTruncated", note_trunc},
      {"spans", spans}
    };
  }

  json run_output(const json &in, const std::atomic<bool> &cancelled) {
    if (in.contains("requiredText") && in["requiredText"].is_string() && !in["requiredText"].get<std::string>().empty())
      return {{"text", in["requiredText"].get<std::string>()}, {"truncated", false}};
    bool short_output = in.value("shortOutput", false);
    std::string sys = short_output
      ? "당신은 책상 위 반려 로봇 DeskPet입니다. 한국어 존댓말로 짧고 핵심만 또렷하게 답하세요. "
        "레시피나 방법·절차를 설명할 때는 긴 본문 대신 핵심 준비물(재료)과 5~6문장 이내의 과정 요약으로만 정리하세요. "
        "음성 합성(TTS)으로 바로 읽히므로 별표(*) 등 마크다운 강조 기호나 괄호() 부가 서술은 절대 사용하지 마세요."
      : "당신은 책상 위 반려 로봇 DeskPet입니다. 한국어로 짧고 친절하게 한두 문장으로 답하세요. "
        "음성 합성(TTS)으로 바로 읽히므로 별표(*) 등 마크다운 강조 기호나 괄호() 부가 서술은 사용하지 마세요.";
    std::vector<std::pair<std::string, std::string>> msgs = {{"system", sys}};
    if (in.contains("messages") && in["messages"].is_array()) {
      for (const auto &m : in["messages"])
        if (m.contains("role") && m.contains("content"))
          msgs.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
    }
    if (msgs.size() == 1 && in.contains("prompt") && in["prompt"].is_string())
      msgs.push_back({"user", in["prompt"].get<std::string>()});
    std::string prompt = render_chatml(msgs, true);
    auto tokens = tokenizer.encode(prompt);

    std::string out;
    bool truncated = true;
    for (int step = 0; step < max_tokens; ++step) {
      if (cancelled.load()) fail("CANCELLED", "output generation cancelled");
#if defined(DESKPET_HAS_TENSORRT)
      auto logits_all = trt_runner->forward_sequence(tokens);
      int v_size = trt_runner->hidden_or_vocab_dim;
      const float *logits = logits_all.data() + (tokens.size() - 1) * static_cast<size_t>(v_size);
      int32_t next = static_cast<int32_t>(std::max_element(logits, logits + v_size) - logits);
#else
      int32_t next = tokenizer.eos_id;
#endif
      if (next == tokenizer.eos_id || next == tokenizer.eot_id) { truncated = false; break; }
      out += tokenizer.piece_of(next);
      tokens.push_back(next);
      if (static_cast<int>(tokens.size()) >= n_ctx) break;
    }
    size_t b = trim_left(out, 0, out.size()), e = trim_right(out, b, out.size());
    return {{"text", out.substr(b, e - b)}, {"truncated", truncated}};
  }

  std::string infer(const json &in, const std::atomic<bool> &cancelled) override {
    return (role_name == "router" ? run_router(in, cancelled) : run_output(in, cancelled)).dump();
  }
};

} // namespace

std::unique_ptr<Model> make_language_model(const json &c) {
  return std::make_unique<TrtLanguageModel>(c);
}

} // namespace deskpet
