# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/
cmake_minimum_required(VERSION 3.14.4)

# Project name
project(application CXX)
include(CMakePrintHelpers) # Helper

# -------------------------------------------------------------------------------------------
# Set PKG_CFLAGS
# ---------------------------------PROJECT_NAME----------------------------------------------------------
set(CMAKE_CXX_COMPILER /usr/bin/c++)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -fPIC)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
# Include dirs
# -------------------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
# -------------------------------------------------------------------------------------------
# 3rd party Dependencies
# -------------------------------------------------------------------------------------------
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
cmake_print_variables(OpenCV_INCLUDE_DIRS)

## TVM Runtime
find_library(TVM_RUNTIME_LIB REQUIRED
            NAMES liblre.so
            PATHS
            /usr/lib/
)
cmake_print_variables(TVM_RUNTIME_LIB)

include_directories(/usr/include/)

include(FetchContent) # once in the project to include the module
FetchContent_Declare(dmlccore
                     GIT_REPOSITORY https://github.com/dmlc/dmlc-core.git
                     GIT_TAG 09511cf9fe5ff103900a5eafb50870dc84cc17c8)
if(NOT dmlccore_POPULATED)
  FetchContent_Populate(dmlccore)
endif()
include_directories(${dmlccore_SOURCE_DIR}/include)

FetchContent_Declare(dlpack
                     GIT_REPOSITORY https://github.com/dmlc/dlpack.git
                     GIT_TAG e2bdd3bee8cb6501558042633fa59144cc8b7f5f)
if(NOT dlpack_POPULATED)
  FetchContent_Populate(dlpack)
endif()
include_directories(${dlpack_SOURCE_DIR}/include)

FetchContent_Declare(json 
                    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
                    DOWNLOAD_EXTRACT_TIMESTAMP true)
FetchContent_MakeAvailable(json)

#TORCH
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Optional find_package for TorchVision
find_package(TorchVision QUIET)

if(TorchVision_FOUND)
  # If TorchVision is found, define a macro to indicate its availability
  add_compile_definitions(HAVE_TORCHVISION)
endif()
if(NOT TorchVision_FOUND)
  message(WARNING "TorchVision not found. Post Processing would be executed on CPU.")
endif()

# -------------------------------------------------------------------------------------------
# Links to target
# -------------------------------------------------------------------------------------------
if(TorchVision_FOUND)
  target_link_libraries(${PROJECT_NAME}
    ${TVM_RUNTIME_LIB}
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    torchvision # Only link if found
    nlohmann_json::nlohmann_json
    -lstdc++fs
  )
else()
  target_link_libraries(${PROJECT_NAME}
    ${TVM_RUNTIME_LIB}
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
    -lstdc++fs
  )
endif()
