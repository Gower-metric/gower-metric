cmake_minimum_required(VERSION 3.15...3.27)
project(gower-metric LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_package(OpenMP)

nanobind_add_module(cpp_engine
  STABLE_ABI
  NB_STATIC
  FREE_THREADED
  LTO
  NOMINSIZE
  MUSL_DYNAMIC_LIBCPP
  gower_metric/cpp/bindings.cpp
  gower_metric/cpp/config/gower_config.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(cpp_engine PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS cpp_engine LIBRARY DESTINATION gower_metric/cpp)

nanobind_add_stub(
  cpp_engine_stub
  MODULE cpp_engine
  OUTPUT cpp_engine.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:cpp_engine>
  DEPENDS cpp_engine
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cpp_engine.pyi"
        DESTINATION gower_metric/cpp)