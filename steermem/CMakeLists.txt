# steermem-cli: the steermem sidecar and protocol mode (steermem/README.md). Configure the tree with
# -DLLAMA_BUILD_STEERMEM=ON and build the steermem-cli target; add -DGGML_CUDA=ON for NVIDIA GPUs.
find_package(Threads REQUIRED)

add_executable(steermem-cli steermem.cpp ${PROJECT_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp)
target_include_directories(steermem-cli PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(steermem-cli PRIVATE llama ggml Threads::Threads)
target_compile_features(steermem-cli PRIVATE cxx_std_17)

if (WIN32)
    target_link_libraries(steermem-cli PRIVATE ws2_32)
    target_compile_definitions(steermem-cli PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()
if (MSVC)
    target_compile_options(steermem-cli PRIVATE /utf-8 /bigobj)
endif()

install(TARGETS steermem-cli RUNTIME)
