# -------------------------------------------------------------------------------------------
# Copyright (C) Latent AI. All rights reserved.
# -------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14.4)

# Project name
project(application CXX)
include(CMakePrintHelpers) # Helper

# -------------------------------------------------------------------------------------------
# Set PKG_CFLAGS
# ---------------------------------PROJECT_NAME----------------------------------------------------------
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_COMPILER /usr/bin/c++)
set(CMAKE_CXX_STANDARD 14)
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
cmake_print_variables(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
# -------------------------------------------------------------------------------------------
# Target Declaration
# -------------------------------------------------------------------------------------------
file(GLOB LRE_SOURCES "src/*.cpp")
cmake_print_variables(LRE_SOURCES)
add_executable(${PROJECT_NAME} application.cpp ${LRE_SOURCES})

# -------------------------------------------------------------------------------------------
# Set ENV
# -------------------------------------------------------------------------------------------
set(TVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../latentai-tvm)
cmake_print_variables(TVM_ROOT)
cmake_print_variables(CMAKE_CURRENT_SOURCE_DIR)
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
cmake_print_variables(OpenCV_INCLUDE_DIRS)

# TVM
set(TVM_INCLUDE_DIR ${TVM_ROOT}/include)
cmake_print_variables(TVM_INCLUDE_DIR)
include_directories(${TVM_INCLUDE_DIR})

# TVM Runtime
set(TVM_RUNTIME_LIB ${TVM_ROOT}/build/libtvm_runtime.so)
if(NOT EXISTS ${TVM_RUNTIME_LIB})
  message(FATAL_ERROR "TVM Runtime library not found at ${TVM_RUNTIME_LIB}")
endif()
cmake_print_variables(TVM_RUNTIME_LIB)
include_directories(TVM_RUNTIME_LIB)

# DLPACK
set(DLPACK_INCLUDE_DIR ${TVM_ROOT}/3rdparty/dlpack/include)
cmake_print_variables(DLPACK_INCLUDE_DIR)
include_directories(${DLPACK_INCLUDE_DIR})

# DMLC
set(DMLC_INCLUDE_DIR ${TVM_ROOT}/3rdparty/dmlc-core/include)
cmake_print_variables(DMLC_INCLUDE_DIR)
include_directories(${DMLC_INCLUDE_DIR})

#TORCH
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS} )

find_package(TorchVision REQUIRED)


# -------------------------------------------------------------------------------------------
# Links to target
# -------------------------------------------------------------------------------------------
target_link_libraries(
  ${PROJECT_NAME}
  ${TVM_RUNTIME_LIB}
  ${OpenCV_LIBS} 
  ${TORCH_LIBRARIES}
  torchvision
  -lstdc++fs
)