cmake_minimum_required(VERSION 3.22)
project(deskpet_product_trt LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DESKPET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../deskpet-pipeline" CACHE PATH "deskpet-pipeline root")
set(DESKPET_NATIVE_SRC "${DESKPET_ROOT}/.build/native-src" CACHE PATH "External C++ source cache")
set(SHERPA_ROOT "${DESKPET_ROOT}/.build/native-deps/sherpa" CACHE PATH "sherpa-onnx SDK root")

# Build ONLY whisper.cpp (CPU STT, ~5 files). llama.cpp is completely excluded.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)

add_subdirectory("${DESKPET_NATIVE_SRC}/whisper" "${CMAKE_CURRENT_BINARY_DIR}/whisper" EXCLUDE_FROM_ALL)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/local/cuda/include /usr/include)
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS /usr/local/cuda/include /usr/include/aarch64-linux-gnu /usr/include)

find_library(NVINFER_LIB nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib)
find_library(NVONNXPARSER_LIB nvonnxparser
  HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib)
find_library(CUDART_LIB cudart
  HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu /usr/lib)

add_library(deskpet_inference SHARED
  "${DESKPET_ROOT}/native/backend.cpp"
  "${DESKPET_ROOT}/native/speech.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/trt_language_model.cpp"
)
target_include_directories(deskpet_inference PRIVATE
  "${DESKPET_ROOT}/Sources/CNativeInference/include"
  "${DESKPET_ROOT}/native"
  "${DESKPET_ROOT}/.build/native-deps/include"
  "${DESKPET_NATIVE_SRC}/llama/vendor"
  "${SHERPA_ROOT}/include"
)
if(TENSORRT_INCLUDE_DIR)
  target_include_directories(deskpet_inference PRIVATE "${TENSORRT_INCLUDE_DIR}")
  target_compile_definitions(deskpet_inference PRIVATE DESKPET_HAS_TENSORRT=1)
endif()
if(CUDA_INCLUDE_DIR)
  target_include_directories(deskpet_inference PRIVATE "${CUDA_INCLUDE_DIR}")
endif()

target_link_directories(deskpet_inference PRIVATE
  "${SHERPA_ROOT}/lib"
  "/usr/local/cuda/lib64"
  "/usr/lib/aarch64-linux-gnu"
)
target_link_libraries(deskpet_inference PRIVATE
  whisper ggml
  sherpa-onnx-c-api onnxruntime
)
if(NVINFER_LIB)
  target_link_libraries(deskpet_inference PRIVATE "${NVINFER_LIB}")
endif()
if(NVONNXPARSER_LIB)
  target_link_libraries(deskpet_inference PRIVATE "${NVONNXPARSER_LIB}")
endif()
if(CUDART_LIB)
  target_link_libraries(deskpet_inference PRIVATE "${CUDART_LIB}")
endif()
if(UNIX AND NOT APPLE)
  target_link_options(deskpet_inference PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/whisper/src:$ORIGIN/whisper/ggml/src:${SHERPA_ROOT}/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu")
endif()
