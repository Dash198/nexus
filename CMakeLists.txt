cmake_minimum_required(VERSION 3.10)
project(nexus_fs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find FUSE3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE REQUIRED fuse3)

# Define the executable
add_executable(nexus src/main.cpp src/stb_image.cpp)

# 1. Target Include Directories (Modern, scoped only to 'nexus')
target_include_directories(nexus PRIVATE
    ${FUSE_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
)

# 2. Tell CMake where the ONNX library files (.so) are for linking
target_link_directories(nexus PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)

# 3. Link the actual libraries
target_link_libraries(nexus PRIVATE
    onnxruntime
    ${FUSE_LIBRARIES}
)

# 4. CRITICAL: Bake the RPATH into the binary so it doesn't crash at runtime
target_link_options(nexus PRIVATE "-Wl,-rpath=${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib")

# 5. FUSE specific compile definitions and options
target_compile_definitions(nexus PUBLIC -D_FILE_OFFSET_BITS=64 -DFUSE_USE_VERSION=31)
target_compile_options(nexus PUBLIC ${FUSE_CFLAGS})
