cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if("$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Targeted CUDA architectures")
endif()

if(MSVC)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Available configuration types to select")
endif()

project(lala-power
  HOMEPAGE_URL "https://github.com/lattice-land/lala-power"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" OFF)
option(LALA_POWER_BUILD_TESTS "LALA_POWER_BUILD_TESTS" OFF)

# Dependencies

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(lala_core SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-core/")
else()
  FetchContent_Declare(
    lala_core
    GIT_REPOSITORY "https://github.com/lattice-land/lala-core.git"
    GIT_TAG        v1.0.4
  )
endif()

FetchContent_MakeAvailable(lala_core)

# Preparing the library

add_library(lala_power INTERFACE)
target_link_libraries(lala_power INTERFACE lala_core)
target_include_directories(lala_power INTERFACE include)

if(LALA_POWER_BUILD_TESTS)

# Google Test dependency

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW 1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

if(LOCAL_DEPS)
  FetchContent_Declare(lala_pc SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-pc/")
  FetchContent_Declare(lala_parsing SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-parsing/")
else()
  FetchContent_Declare(
    lala_pc
    GIT_REPOSITORY "https://github.com/lattice-land/lala-pc.git"
    GIT_TAG        v1.0.4
  )

  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.0.3
  )
endif()

FetchContent_MakeAvailable(lala_pc lala_parsing)

# I. CPU Tests (ending with "_test.cpp")
file(GLOB cpu_test_files tests/*_test.cpp)
foreach(file ${cpu_test_files})
  cmake_path(GET file STEM test_name)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} lala_pc lala_parsing lala_power gtest_main)
  target_include_directories(${test_name} PRIVATE "${lala-core_SOURCE_DIR}/tests/include")
  gtest_discover_tests(${test_name})
endforeach()

endif()

# Documentation

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "Lattice Land Powerdomains Library")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)
set(DOXYGEN_EXCLUDE_SYMBOLS = impl)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_lala_power
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
