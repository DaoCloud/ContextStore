cmake_minimum_required(VERSION 3.20)
project(contextstore_nixl_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CONTEXTSTORE_NIXL_STRICT "Treat warnings as errors" OFF)
set(NIXL_PREFIX "" CACHE PATH "Installed NIXL prefix")
set(NIXL_SOURCE_DIR "" CACHE PATH "NIXL source checkout")
set(NIXL_LIBRARY_DIR "" CACHE PATH "Directory containing NIXL shared libraries")

add_library(plugin_CONTEXTSTORE SHARED
  contextstore_backend.cpp
  contextstore_plugin.cpp
)

target_include_directories(plugin_CONTEXTSTORE
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NIXL_SOURCE_DIR)
  target_include_directories(plugin_CONTEXTSTORE SYSTEM
    PRIVATE
      ${NIXL_SOURCE_DIR}/src/api/cpp
      ${NIXL_SOURCE_DIR}/src/api/cpp/backend
      ${NIXL_SOURCE_DIR}/src/core/telemetry
      ${NIXL_SOURCE_DIR}/src/utils
  )
elseif(NIXL_PREFIX)
  target_include_directories(plugin_CONTEXTSTORE SYSTEM
    PRIVATE
      ${NIXL_PREFIX}/include
      ${NIXL_PREFIX}/include/nixl
      ${NIXL_PREFIX}/include/nixl/backend
  )
  target_link_directories(plugin_CONTEXTSTORE
    PRIVATE
      ${NIXL_PREFIX}/lib
      ${NIXL_PREFIX}/lib64
  )
  if(NOT NIXL_LIBRARY_DIR)
    set(NIXL_LIBRARY_DIR ${NIXL_PREFIX}/lib)
  endif()
else()
  message(FATAL_ERROR "Set either -DNIXL_SOURCE_DIR=/path/to/nixl or -DNIXL_PREFIX=/path/to/nixl/prefix")
endif()

if(NIXL_LIBRARY_DIR)
  target_link_directories(plugin_CONTEXTSTORE PRIVATE ${NIXL_LIBRARY_DIR})
  target_link_libraries(plugin_CONTEXTSTORE PRIVATE
    -Wl,--no-as-needed
    nixl_build
    nixl
    nixl_common
    -Wl,--as-needed
  )
  set_target_properties(plugin_CONTEXTSTORE PROPERTIES
    BUILD_RPATH "${NIXL_LIBRARY_DIR}"
    INSTALL_RPATH "${NIXL_LIBRARY_DIR}"
  )
endif()

target_link_libraries(plugin_CONTEXTSTORE PRIVATE dl)

if(CONTEXTSTORE_NIXL_STRICT)
  target_compile_options(plugin_CONTEXTSTORE PRIVATE -Wall -Wextra -Werror)
else()
  target_compile_options(plugin_CONTEXTSTORE PRIVATE -Wall -Wextra)
endif()

set_target_properties(plugin_CONTEXTSTORE PROPERTIES
  OUTPUT_NAME "plugin_CONTEXTSTORE"
  PREFIX "lib"
)

install(TARGETS plugin_CONTEXTSTORE LIBRARY DESTINATION lib/nixl/plugins)
install(FILES contextstore_nixl_client.h DESTINATION include/contextstore)
