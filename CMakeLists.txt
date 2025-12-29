cmake_minimum_required (VERSION 3.29)

project (BaseLine)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

# Add option to enable/disable C++ 20 module
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

# Enable C++ module dependency scanning only if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

add_subdirectory(vma_raii)

find_package (glfw3 REQUIRED)
find_package (glm REQUIRED)
find_package (Vulkan REQUIRED) # using 1.4 at the moment
find_package (tinyobjloader REQUIRED)
find_package (VulkanMemoryAllocator CONFIG REQUIRED)

# set up Vulkan C++ module only if enabled
if(ENABLE_CPP20_MODULE)
  add_library(VulkanCppModule)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)

  target_compile_definitions(VulkanCppModule
          PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
  target_include_directories(VulkanCppModule
          PUBLIC
          "${Vulkan_INCLUDE_DIR}"
  )
  target_link_libraries(VulkanCppModule
          PUBLIC
          Vulkan::Vulkan
  )

  set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

  # Add MSVC-specific compiler options for proper C++ module support
  if(MSVC)
    target_compile_options(VulkanCppModule PRIVATE
      /std:c++latest      # Use latest C++ standard for better module support
      /permissive-        # Standards conformance mode
      /Zc:__cplusplus     # Enable correct __cplusplus macro
      /EHsc               # Enable C++ exception handling
      /Zc:preprocessor    # Use conforming preprocessor
      /translateInclude   # Automatically translate #include to import for standard library
    )
  endif()

  target_sources(VulkanCppModule
          PUBLIC
          FILE_SET cxx_modules TYPE CXX_MODULES
          BASE_DIRS
          "${Vulkan_INCLUDE_DIR}"
          FILES
          "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )
else()
  # Create a dummy interface library when C++ 20 module is disabled
  add_library(VulkanCppModule INTERFACE)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
  target_compile_definitions(VulkanCppModule
          INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
endif()

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

add_executable (glslang::validator IMPORTED)
find_program (GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set_property (TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "APP_NAME" "SOURCES" ${ARGN})
  set (SHADERS_DIR "${CMAKE_BINARY_DIR}/${SHADER_APP_NAME}/shaders")

  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )

  set(SPV_OUTPUTS "")
  foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME_WE ${SHADER_SOURCE} NAME_WE)
    set(SPV_OUTPUT_FILENAME ${SHADER_NAME_WE}.spv)
    set(SPV_OUTPUT_FULLPATH "${SHADERS_DIR}/${SPV_OUTPUT_FILENAME}")
    list(APPEND SPV_OUTPUTS ${SPV_OUTPUT_FULLPATH})

    set (ENTRY_POINTS -entry vertMain -entry fragMain)
    file(STRINGS "${SHADER_SOURCE}" COMPUTE_PRESENT REGEX "compMain")
    if(COMPUTE_PRESENT)
      list(APPEND ENTRY_POINTS -entry compMain)
    endif()

    add_custom_command (
            OUTPUT  ${SPV_OUTPUT_FULLPATH}
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCE} -target spirv -profile spirv_1_4+spvRayQueryKHR -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${SPV_OUTPUT_FILENAME}
            WORKING_DIRECTORY ${SHADERS_DIR}
            DEPENDS ${SHADER_SOURCE} ${SHADERS_DIR}
            COMMENT "Compiling Slang Shader: ${SHADER_SOURCE}"
            VERBATIM
    )
  endforeach()

  add_custom_target (${TARGET} DEPENDS ${SPV_OUTPUTS})
endfunction()

function (add_app APP_NAME)
  cmake_parse_arguments (APP "" "" "SHADERS;LIBS;TEXTURES;MODELS" ${ARGN})
  file(GLOB ALL_SOURCES "src/*.cpp" "src/*.h" "src/*.hpp")
  add_executable (${APP_NAME} ${ALL_SOURCES})
  target_compile_definitions(${APP_NAME}
  PRIVATE
    SHADER_DIR="${CMAKE_BINARY_DIR}/${APP_NAME}/shaders"
    APP_NAME="${APP_NAME}"
  )
  set_target_properties (${APP_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${APP_NAME})
  set_target_properties (${APP_NAME} PROPERTIES CXX_STANDARD 20)
  target_link_libraries (${APP_NAME} Vulkan::cppm glfw GPUOpen::VulkanMemoryAllocator)
  target_include_directories (${APP_NAME} 
  PRIVATE
    ${STB_INCLUDEDIR}
    src/
  )

  # Add compile definition if C++ 20 module is enabled
  if(ENABLE_CPP20_MODULE)
    target_compile_definitions(${APP_NAME} PRIVATE USE_CPP20_MODULES=1)

    # Add MSVC-specific compiler options for proper C++ module support
    # These flags need to be consistent between the module and its consumers.
    if(MSVC)
      target_compile_options(${APP_NAME} PRIVATE
        /std:c++latest      # Use latest C++ standard for better module support
        /permissive-        # Standards conformance mode
        /Zc:__cplusplus     # Enable correct __cplusplus macro
        /EHsc               # Enable C++ exception handling
        /Zc:preprocessor    # Use conforming preprocessor
        /translateInclude   # Automatically translate #include to import for standard library
      )
    endif()
  endif()

  if(WIN32)
    target_compile_definitions(${APP_NAME} PRIVATE
      VK_USE_PLATFORM_WIN32_KHR)
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
      set_target_properties(${APP_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${APP_NAME}")
    endif()
  endif()

  if (DEFINED APP_SHADERS)
    set (APP_SHADER_SLANG_TARGET ${APP_NAME}_slang_shader)
    set(SHADER_SLANG_SOURCES "")
    foreach(SHADER_NAME IN LISTS APP_SHADERS)
      list(APPEND SHADER_SLANG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/${SHADER_NAME}.slang")
    endforeach()
    if(SHADER_SLANG_SOURCES)
      add_slang_shader_target( ${APP_SHADER_SLANG_TARGET} APP_NAME ${APP_NAME} SOURCES ${SHADER_SLANG_SOURCES})
      add_dependencies(${APP_NAME} ${APP_SHADER_SLANG_TARGET})
    endif()
  endif ()
  if (DEFINED APP_LIBS)
    target_link_libraries (${APP_NAME} ${APP_LIBS})
  endif ()
  if (DEFINED APP_MODELS)
    list(TRANSFORM APP_MODELS PREPEND "${CMAKE_SOURCE_DIR}/assets/")
    file (COPY ${APP_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/${APP_NAME}/models)
  endif ()
  if (DEFINED APP_TEXTURES)
    list(TRANSFORM APP_TEXTURES PREPEND "${CMAKE_SOURCE_DIR}/assets/")
    file (COPY ${APP_TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/${APP_NAME}/textures)
  endif ()
endfunction ()

add_app (app1
  SHADERS base_shader original
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader vma_raii)