# DeskPet Product

Jetson Orin Nano에서 DeskPet을 구성하고 실행하는 배포 진입점입니다. 이 저장소의 `product.sh`가 한국어 음성 모델과 네이티브 추론 라이브러리를 준비하고, 장치 데몬·Harness·Swift 파이프라인을 같은 기기에서 실행합니다. 업무 연동과 로컬 음성 처리를 함께 시험하는 데 필요한 설정과 구동 순서를 이곳에서 관리합니다.

## 배포 구성

네 저장소를 하나의 상위 디렉터리 아래에 둡니다. 런처는 아래 이름으로 형제 폴더를 찾고 빠진 저장소를 복제합니다. 모델과 서비스 설정을 미리 넣으려면 필요한 저장소를 먼저 복제하세요.

| 폴더 | 실행 책임 | 로컬 설정 |
| --- | --- | --- |
| `deskpet-product/` | 저장소·런타임·모델 준비 및 통합 시작 | `product.sh` 인자 |
| `deskpet-pipeline/` | 음성 처리, 라우터와 출력 모델 구동, 업무 요청·장치 연결 | `Models/native.jetson.local.json` |
| `deskpet-hardware/` | 오디오·LCD·터치·귀 구동과 Unix 소켓 제공 | `config/jetson.json` |
| `deskpet-harness/` | Eureka·GitHub, LLM/도구 호출, 상태 기록과 웹 화면 | `.env`, SQLite 경로 |

## 준비

**대상 환경**은 NVIDIA Jetson Orin Nano 8GB, JetPack 6.x 계열의 Ubuntu와 CUDA입니다. Git·curl·unzip·CMake·C/C++ 빌드 도구, Node.js 22.13 이상과 pnpm이 필요합니다. Swift 또는 Rust가 없으면 스크립트가 설치를 시도합니다. 첫 실행에는 런타임·모델 다운로드 및 네이티브 빌드가 있습니다.

**모델 파일**은 파이프라인의 `Models/` 아래에 둡니다. 공개 모델은 스크립트가 내려받고 파일이 있으면 재사용합니다. 프로젝트에서 학습·변환한 라우터와 감정 모델은 별도로 준비합니다.

| 준비할 모델 | 경로 또는 항목 |
| --- | --- |
| 라우터 백본과 헤드 | `Models/router/backbone-merged-q4_0.gguf`, `Models/router/heads-f16.gguf` |
| 라우터 설정과 채팅 템플릿 | `Models/router/router_config.json`, `Models/router/tokenizer/chat_template.jinja` |
| 감정 분류 그래프 | `Models/emotion/model.onnx` |

[모델 가중치 준비](https://github.com/hansung-capstone-team-8-new-line/deskpet-pipeline/blob/main/Models/README.md)

**Harness 설정**은 `deskpet-harness/.env`에서 지정합니다. `HARNESS_WS_TOKEN`, `GITHUB_TOKEN`, 데모 대상인 `GITHUB_TEST_REPO`와 `GITHUB_TEST_PR`가 필요합니다. Eureka와 LLM 연동 값은 사용할 서비스에 맞춰 채웁니다. 쓰기 실행에는 SQLite 경로인 `DESKPET_STORE_PATH`가 필요합니다. [환경 변수 예제](https://github.com/hansung-capstone-team-8-new-line/deskpet-harness/blob/app/.env.example)

**실물 장치**는 LCD·USB 오디오·PCA9685를 연결하고, 설치 안내에 따라 장치 ID와 I²C 버스를 확인하고 귀를 보정합니다. 제품 런처의 실물 모드는 하드웨어 체크아웃의 `config/jetson.json`을 읽으므로 거기에 해당 값을 지정합니다. [Jetson 장치 설치](https://github.com/hansung-capstone-team-8-new-line/deskpet-hardware/blob/main/docs/jetson-setup.md)

## 실행

```sh
# 구성과 모델 경로 확인
./product.sh --check

# 실제 장치 사용
./product.sh --real-hardware

# 저메모리 음성 모델 사용
./product.sh --real-hardware --low-memory
```

`--check`는 런처의 모델·설정 확인에서 멈춥니다. 그 전에 `product.sh`의 다운로드·빌드 준비 단계가 실행될 수 있습니다. 최초 실물 실행 전에 별도 모드로 빌드한 데몬이 있으면 Linux 드라이버를 포함하도록 다시 구성하세요. 제품 스크립트는 기존 실행 파일을 재사용할 수 있습니다.

| 옵션 | 동작 |
| --- | --- |
| `--real-hardware` | 실제 ALSA·I²C 장치와 Jetson 설정으로 실행 |
| `--keyboard` | 음성 인식 대신 터미널에서 문장을 입력 |
| `--headless` | 그래픽 창을 띄우지 않고 실행 |
| `--low-memory` | 아래 표의 경량 음성 모델 조합 사용 |
| `--use-audio-gpu` | STT·TTS 추론도 파이프라인의 GPU 실행기를 거치도록 설정 |
| `--no-build` | 이미 있는 빌드 산출물을 우선 재사용. 필수 산출물이 없으면 제품 스크립트가 빌드할 수 있음 |

## 음성 모델 프로필

| 프로필 | 음성 인식 | 음성 합성 |
| --- | --- | --- |
| 기본 | sherpa-onnx 한국어 Zipformer | ONNX Runtime Qwen3-TTS 0.6B CustomVoice. 준비된 INT4 그래프 우선 |
| `--low-memory` | 한국어 스트리밍 Zipformer 174M INT8 | ONNX Runtime MeloTTS 한국어 모델. 다운로드 기본값은 FP16 |

두 프로필에서 VAD는 Silero, 라우터와 출력 모델은 `llama.cpp` CUDA 백엔드입니다. `configure_jetson_json.swift`가 로컬 설정의 모델 경로와 백엔드를 채우고 필요한 경우 원본 Qwen3-TTS 그래프에서 배포용 양자화 파일을 만듭니다. [설정 생성기](configure_jetson_json.swift)

## 런처가 하는 일

1. 필요한 Swift/Rust 및 모델 파일을 준비하고 Jetson 로컬 설정을 생성하거나 갱신합니다.
2. C++ 추론 플러그인 `.build/native/libdeskpet_inference.so`와 장치 데몬을 준비하고, Harness 의존성과 Swift 실행 파일을 빌드합니다.
3. Jetson 런처를 통해 장치 소켓과 Harness 연결을 준비한 뒤 사용자 입력을 받습니다. 실행별 서비스 로그는 파이프라인의 `.build/run/` 아래에 남고 모델 호출 기록은 `.build/performance/`에서 확인할 수 있습니다.
4. `Ctrl-C`로 해당 실행이 시작한 프로세스를 종료합니다. 키보드 모드에서는 `/quit`으로도 종료할 수 있습니다.

| 관련 저장소 | 찾을 수 있는 내용 |
| --- | --- |
| [DeskPet Pipeline](https://github.com/hansung-capstone-team-8-new-line/deskpet-pipeline) | 입력·의도 분기, 로컬과 외부 실행, 모델 준비와 관측 |
| [DeskPet Hardware](https://github.com/hansung-capstone-team-8-new-line/deskpet-hardware) | 실물 구성, 보정, 소켓 계약, 조립 원본 |
| [DeskPet Harness](https://github.com/hansung-capstone-team-8-new-line/deskpet-harness) | 업무 연동, 영속 기록, 대시보드 |
| [DeskPet Knowledge](https://github.com/hansung-capstone-team-8-new-line/knowledge) | 시스템 설계, 통합 인터페이스, 검증 시나리오 |
