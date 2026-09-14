"""fork edit 0022 (2026-09-14): steermem-cli as a CMake target, so every platform builds it the same way.

Until now steermem-cli was linked by hand (g++ on Linux, the MinGW cross-compiler for Windows). llama.cpp's Windows
CUDA builds need MSVC and nvcc, which only CMake drives cleanly, so the root CMakeLists.txt gains an option that adds
steermem/ (its CMakeLists.txt is code/llama/fork/steermem_CMakeLists.txt):

  cmake -B build -DLLAMA_BUILD_STEERMEM=ON [-DGGML_CUDA=ON] && cmake --build build --target steermem-cli

  python 0022_cmake_target.py path/to/llama.cpp/CMakeLists.txt
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
if "LLAMA_BUILD_STEERMEM" in s:
    raise SystemExit("0022 is already applied to %s" % p)
old = 'option(LLAMA_BUILD_MTMD "llama: build tools/mtmd library standalone" OFF)'
assert s.count(old) == 1, "anchor: %d matches" % s.count(old)
s = s.replace(old, '''# steermem (fork edit 0022): the steermem-cli target, off by default
option(LLAMA_BUILD_STEERMEM "llama: build steermem-cli (steermem/)" OFF)
if (LLAMA_BUILD_STEERMEM)
    add_subdirectory(steermem)
endif()

''' + old)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0022 applied to %s" % p)
