# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

# This only builds for x86-64/aarch64 because that's all we have coded in here.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)" AND
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  message(STATUS "IREE custom_dispatch/cpu/plugin ignored -- only builds for x86_64/aarch64 (today)")
  return()
endif()

# TODO(benvanik): make these separable - need to plumb environment into the
# lit test so we know which one to test.
if(NOT IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY OR
   NOT IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)
  message(STATUS "IREE custom_dispatch/cpu/plugin ignored -- need all plugin mechanisms (today)")
  return()
endif()

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

add_library(iree_samples_custom_dispatch_cpu_system_plugin SHARED
  system_plugin.c
)
target_include_directories(iree_samples_custom_dispatch_cpu_system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)
# TODO(benvanik): use native platform extension via CMAKE_SHARED_LIBRARY_SUFFIX.
# Need to have environment variables on lit test suite to get it into lit.
set_target_properties(iree_samples_custom_dispatch_cpu_system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "system_plugin"
    SUFFIX ""
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

# embedded-elf plugin mechanism for standalone portable ELFs.
# We use the same flags we do in the compiler for our embedded elfs found under
# compiler/src/iree/compiler/Dialect/HAL/Target/LLVMCPU/internal/EmbeddedLinkerTool.cpp
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)

function(standalone_plugin_library _ARCH)
set(_NAME iree_samples_custom_dispatch_cpu_standalone_plugin_${_ARCH})
add_custom_command(
  OUTPUT
    standalone_plugin_${_ARCH}.o
  DEPENDS
    standalone_plugin.c
    ${IREE_CLANG_TARGET}
  COMMAND ${IREE_CLANG_TARGET}
    -target ${_ARCH}-unknown-unknown-eabi-elf
    -isystem ${IREE_BINARY_DIR}/third_party/llvm-project/llvm/lib/clang/17/include
    -std=c17
    -fPIC
    -ffreestanding
    -fvisibility=hidden
    -fno-plt
    -fno-rtti
    -fno-exceptions
    -fdata-sections
    -ffunction-sections
    -funique-section-names
    -I ${IREE_SOURCE_DIR}/runtime/src/
    -c ${CMAKE_CURRENT_SOURCE_DIR}/standalone_plugin.c
    -o ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.o
  VERBATIM
)
add_custom_command(
  OUTPUT
    standalone_plugin_${_ARCH}.so
  DEPENDS
    standalone_plugin_${_ARCH}.o
    ${IREE_LLD_TARGET}
  COMMAND ${IREE_LLD_TARGET}
    -flavor gnu
    --build-id=none
    -nostdlib
    -static
    -shared
    --no-undefined
    --no-allow-shlib-undefined
    --allow-multiple-definition
    --gc-sections
    -z now
    -z relro
    --discard-all
    --icf=all
    --ignore-data-address-equality
    --ignore-function-address-equality
    --hash-style=sysv
    --strip-debug
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.o
    -o ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.so
  VERBATIM
)
add_custom_target(${_NAME} DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.so
)
endfunction()

# Build the standalone_plugin_*.so files for each architecture we target.
standalone_plugin_library(aarch64)
standalone_plugin_library(x86_64)

add_custom_command(
  OUTPUT
    standalone_plugin.sos
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_aarch64.so
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_x86_64.so
    iree-fatelf
  COMMAND iree-fatelf join
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_aarch64.so
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_x86_64.so
    > ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin.sos
  VERBATIM
)
add_custom_target(iree_samples_custom_dispatch_cpu_standalone_plugin DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin.sos
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)

add_custom_target(iree_samples_custom_dispatch_cpu_plugins DEPENDS
  iree_samples_custom_dispatch_cpu_standalone_plugin
  iree_samples_custom_dispatch_cpu_system_plugin
)
add_dependencies(iree-sample-deps iree_samples_custom_dispatch_cpu_plugins)

iree_lit_test_suite(
  NAME
    examples
  SRCS
    "example.mlir"
  DATA
    standalone_plugin.sos
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)
