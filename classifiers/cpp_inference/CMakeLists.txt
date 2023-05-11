# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

cmake_minimum_required(VERSION 3.10.2)

# Project name
project(application CXX)
include(CMakePrintHelpers) # Helper

# -------------------------------------------------------------------------------------------
# Set PKG_CFLAGS
# ---------------------------------PROJECT_NAME----------------------------------------------------------
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_COMPILER /usr/bin/c++)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -fPIC)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------------------------------
# Set Common CPP Flags
# -------------------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CppFlags)

# -------------------------------------------------------------------------------------------
# Output Directories
# -------------------------------------------------------------------------------------------
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin)

# -------------------------------------------------------------------------------------------
# Target Declaration
# -------------------------------------------------------------------------------------------
file(GLOB LRE_SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} application.cpp ${LRE_SOURCES})

# -------------------------------------------------------------------------------------------
# Include dirs
# -------------------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${LRE_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../utils)

# -------------------------------------------------------------------------------------------
# 3rd party Dependencies
# -------------------------------------------------------------------------------------------
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# # TVM Runtime
find_library(TVM_RUNTIME_LIB REQUIRED
  NAMES libtvm_runtime.so
  PATHS
  /usr/local/bin/
  /usr/tvm/build/
)
cmake_print_variables(TVM_RUNTIME_LIB)

include_directories(/usr/tvm/include/)
include_directories(/usr/local/include/)

include(FetchContent) # once in the project to include the module
FetchContent_Declare(dmlccore
  GIT_REPOSITORY https://github.com/dmlc/dmlc-core.git
  GIT_TAG 21cc7de0dc9fd6acb796e1be6181fa8e6b6c8f41)

if(NOT dmlccore_POPULATED)
  FetchContent_Populate(dmlccore)
endif()

include_directories(${dmlccore_SOURCE_DIR}/include)

FetchContent_Declare(dlpack
  GIT_REPOSITORY https://github.com/dmlc/dlpack.git
  GIT_TAG ddeb264880a1fa7e7be238ab3901a810324fbe5f)

if(NOT dlpack_POPULATED)
  FetchContent_Populate(dlpack)
endif()

include_directories(${dlpack_SOURCE_DIR}/include)

# -------------------------------------------------------------------------------------------
# Links to target
# -------------------------------------------------------------------------------------------
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${TVM_RUNTIME_LIB}
)