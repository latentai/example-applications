# -------------------------------------------------------------------------------------------
# Copyright (C) Latent AI. All rights reserved.
# -------------------------------------------------------------------------------------------
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
# Set ENV
# -------------------------------------------------------------------------------------------
set(TVM_ROOT $ENV{TVM_HOME})

# -------------------------------------------------------------------------------------------
# Include dirs
# -------------------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${LRE_SOURCES})

# -------------------------------------------------------------------------------------------
# 3rd party Dependencies
# -------------------------------------------------------------------------------------------
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TVM
set(TVM_INCLUDE_DIR ${TVM_ROOT}/include)
include_directories(${TVM_INCLUDE_DIR})

# TVM Runtime
set(TVM_RUNTIME_LIB ${TVM_ROOT}/build/libtvm_runtime.so)
include_directories(TVM_RUNTIME_LIB)

# DLPACK
set(DLPACK_INCLUDE_DIR ${TVM_ROOT}/3rdparty/dlpack/include)
include_directories(${DLPACK_INCLUDE_DIR})

# DMLC
set(DMLC_INCLUDE_DIR ${TVM_ROOT}/3rdparty/dmlc-core/include)
include_directories(${DMLC_INCLUDE_DIR})

# -------------------------------------------------------------------------------------------
# Links to target
# -------------------------------------------------------------------------------------------
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${TVM_RUNTIME_LIB}
)