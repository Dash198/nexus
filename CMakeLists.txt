cmake_minimum_required(VERSION 3.10)
project(nexus_fs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find FUSE3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE REQUIRED fuse3)

# 1. Force the Include Directories (HIGHEST PRIORITY)
# This ensures we look in /usr/include/fuse3 before /usr/include/fuse
include_directories(BEFORE ${FUSE_INCLUDE_DIRS})

add_executable(nexus src/main.cpp)

# 1. Tell CMake where the ONNX headers are
target_include_directories(nexus PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)

# 2. Tell CMake where the ONNX library files (.so) are
target_link_directories(nexus PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)

# 3. Link BOTH the ONNX runtime and FUSE libraries to your executable
target_link_libraries(nexus PRIVATE
    onnxruntime
    ${FUSE_LIBRARIES}
)

# 3. CRITICAL: Add the required compile definitions
# FUSE requires these to work on 64-bit Linux
target_compile_definitions(nexus PUBLIC -D_FILE_OFFSET_BITS=64 -DFUSE_USE_VERSION=31)

# Add standard FUSE flags
target_compile_options(nexus PUBLIC ${FUSE_CFLAGS})
