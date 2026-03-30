cmake_minimum_required(VERSION 3.14)
project(rabitq VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------
# Compiler flags: match RaBitQ reference (-Ofast -march=core-avx2)
# --------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=core-avx2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=core-avx2 -fsanitize=address")

# --------------------------------------------------------------------------
# Eigen 3.4.0 via FetchContent
# --------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# --------------------------------------------------------------------------
# Google Test via FetchContent
# --------------------------------------------------------------------------
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

# --------------------------------------------------------------------------
# Library: rabitq (header-only for now, will add .cpp files later)
# --------------------------------------------------------------------------
add_library(rabitq INTERFACE)
target_include_directories(rabitq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rabitq INTERFACE Eigen3::Eigen)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
add_executable(test_setup tests/test_setup.cpp)
target_link_libraries(test_setup PRIVATE rabitq GTest::gtest_main)
target_compile_options(test_setup PRIVATE -mavx2)

include(GoogleTest)
gtest_discover_tests(test_setup)
