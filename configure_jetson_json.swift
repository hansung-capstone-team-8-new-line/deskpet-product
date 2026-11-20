import Foundation

struct ModelConfig: Codable {
    var backend: String
    var role: String
    var device: String
    var paths: [String: String]
    var library: String?
    var contextTokens: Int?
    var maxTokens: Int?
    var threads: Int?
}

struct JetsonConfig: Codable {
    var library: String
    var harnessEnvFile: String?
    var hardwareSocket: String?
    var routerTimeout: Double?
    var outputTimeout: Double?
    var sttTimeout: Double?
    var ttsTimeout: Double?
    var routeConfidenceThreshold: Double?
    var sttConfidenceThreshold: Double?
    var routerContextEnabled: Bool?
    var directLLM: Bool?
    var shortOutput: Bool?
    var performanceDirectory: String?
    var startupReminder: Bool?
    var startupReminderText: String?
    var models: [String: ModelConfig]
}

let args = CommandLine.arguments
let useAudioGpu = args.contains("--use-audio-gpu") || ProcessInfo.processInfo.environment["DESKPET_USE_AUDIO_GPU"] == "1"
let configPath = "Models/native.jetson.local.json"
let files = FileManager.default

guard files.fileExists(atPath: configPath),
      let data = files.contents(atPath: configPath) else {
    fputs("[configure_jetson_json.swift] Failed to read \(configPath)\n", stderr)
    exit(1)
}

var cfg = try JSONDecoder().decode(JetsonConfig.self, from: data)
cfg.library = "../.build/native/libdeskpet_inference.so"
cfg.harnessEnvFile = "../../deskpet-harness/.env"

if var router = cfg.models["router"] {
    router.backend = "llama.cpp"
    router.device = "cuda"
    router.library = nil
    router.paths = [
        "model": "router/backbone-merged-q4_0.gguf",
        "heads": "router/heads-f16.gguf",
        "config": "router/router_config.json",
        "chat_template": "router/tokenizer/chat_template.jinja",
    ]
    if router.contextTokens == nil || router.contextTokens == 0 { router.contextTokens = 2560 }
    if router.maxTokens == nil || router.maxTokens == 0 { router.maxTokens = 128 }
    if router.threads == nil || router.threads == 0 { router.threads = 2 }
    cfg.models["router"] = router
}

if var output = cfg.models["output"] {
    output.backend = "llama.cpp"
    output.device = "cuda"
    output.library = nil
    output.paths = [
        "model": "output/qwen-2b-q4_0.gguf",
    ]
    if output.contextTokens == nil || output.contextTokens == 0 { output.contextTokens = 2048 }
    if output.maxTokens == nil || output.maxTokens == 0 { output.maxTokens = 128 }
    if output.threads == nil || output.threads == 0 { output.threads = 2 }
    cfg.models["output"] = output
}

for role in ["vad", "emotion"] {
    if var model = cfg.models[role] {
        model.library = nil
        cfg.models[role] = model
    }
}

func firstExistingRelative(in subdirs: [String], candidates: [String]) -> String? {
    for dir in subdirs {
        for name in candidates {
            let rel = dir + "/" + name
            if files.fileExists(atPath: "Models/" + rel) {
                return rel
            }
        }
    }
    return nil
}

func matchingDirs(under parent: String, prefix: String) -> [String] {
    guard let entries = try? files.contentsOfDirectory(atPath: "Models/" + parent) else { return [] }
    return entries.filter { $0.lowercased().hasPrefix(prefix) }.sorted().map { parent + "/" + $0 }
}

let audioDevice = useAudioGpu ? "cuda" : "cpu"
let sttDirs = matchingDirs(under: "stt", prefix: "sherpa-onnx-qwen3-asr")
    + matchingDirs(under: "stt", prefix: "qwen3-asr")
    + ["stt/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"]
if var stt = cfg.models["stt"] {
    stt.backend = "sherpa-onnx"
    stt.device = audioDevice
    stt.library = nil
    stt.contextTokens = stt.contextTokens == 0 ? 512 : stt.contextTokens
    stt.maxTokens = stt.maxTokens == 0 ? 128 : stt.maxTokens
    let baseDir = sttDirs.first(where: { files.fileExists(atPath: "Models/" + $0 + "/conv_frontend.onnx") })
        ?? "stt/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    stt.paths = [
        "conv_frontend": baseDir + "/conv_frontend.onnx",
        "encoder": firstExistingRelative(in: [baseDir], candidates: ["encoder.int8.onnx", "encoder.onnx"]) ?? (baseDir + "/encoder.int8.onnx"),
        "decoder": firstExistingRelative(in: [baseDir], candidates: ["decoder.int8.onnx", "decoder.onnx"]) ?? (baseDir + "/decoder.int8.onnx"),
        "tokenizer": baseDir + "/tokenizer",
    ]
    cfg.models["stt"] = stt
}

let ttsDirs = matchingDirs(under: "tts", prefix: "qwen3-tts") + ["tts/qwen3-tts-12hz-0.6b-customvoice"]
if var tts = cfg.models["tts"] {
    tts.backend = "tensorrt"
    tts.device = "cuda"
    tts.library = nil
    tts.maxTokens = tts.maxTokens == 0 ? 2048 : tts.maxTokens
    let baseDir = ttsDirs.first(where: {
        files.fileExists(atPath: "Models/" + $0 + "/config.json")
    }) ?? "tts/qwen3-tts-12hz-0.6b-customvoice"
    let precisionDirs = [baseDir, baseDir + "/cuda_fp16", baseDir + "/cpu_fp32", baseDir + "/cuda_fp32"]
    func ttsArt(_ candidates: [String], fallback: String) -> String {
        firstExistingRelative(in: precisionDirs, candidates: candidates) ?? (baseDir + "/" + fallback)
    }
    var paths: [String: String] = [
        "text_embed": ttsArt(["text_embed.engine", "text_embed.onnx"], fallback: "text_embed.engine"),
        "codec_embed": ttsArt(["codec_embed.engine", "codec_embed.onnx"], fallback: "codec_embed.engine"),
        "residual_embed": ttsArt(["residual_embed.engine", "residual_embed.onnx"], fallback: "residual_embed.engine"),
        "talker": ttsArt(["talker_cache.engine", "talker.engine", "talker_cache.onnx", "talker.onnx"], fallback: "talker_cache.engine"),
        "code_predictor": ttsArt(["code_predictor.engine", "code_predictor.onnx"], fallback: "code_predictor.engine"),
        "tok_decoder": ttsArt(["tok_decoder.engine", "tok_decoder.onnx"], fallback: "tok_decoder.engine"),
        "config": baseDir + "/config.json",
    ]
    if let tokJson = firstExistingRelative(in: [baseDir], candidates: ["tokenizer.json"]) {
        paths["tokenizer"] = tokJson
    } else {
        paths["vocab"] = baseDir + "/vocab.json"
        paths["merges"] = baseDir + "/merges.txt"
    }
    tts.paths = paths
    cfg.models["tts"] = tts
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
var outData = try encoder.encode(cfg)
outData.append(0x0A)
try outData.write(to: URL(fileURLWithPath: configPath), options: .atomic)
