CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(rela)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -Wfatal-errors")

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")

# ── pybind11: prefer submodule, fall back to find_package ────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
else()
  find_package(pybind11 CONFIG REQUIRED)
endif()

# ── find pytorch ──────────────────────────────────────────────────────────────
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

# Resolve the Python-binding torch library (name differs by OS)
if(APPLE)
  set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.dylib")
else()
  set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")
endif()

message("---------------------")
message("TorchPath    : ${TorchPath}")
message("PYBIND_FLAGS : ${PYBIND_FLAGS}")
message("CXX_FLAGS    : ${CMAKE_CXX_FLAGS}")
message("libtorch_py  : ${TORCH_PYTHON_LIBRARIES}")
message("---------------------")

# ── rela_lib (shared by pybind module) ────────────────────────────────────────
add_library(rela_lib
  rela/transition.cc
  rela/replay.cc
  rela/concurrent_queue.cc
  rela/tensor_dict.cc
  rela/episode.cc
)
target_include_directories(rela_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

# ── rela Python extension ─────────────────────────────────────────────────────
pybind11_add_module(rela rela/pybind.cc)
target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
