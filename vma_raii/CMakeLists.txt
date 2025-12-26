add_library(vma_raii)
set_target_properties (vma_raii PROPERTIES CXX_STANDARD 20)

target_sources(vma_raii
    PRIVATE
    vma_raii.cpp
    vma_raii.h
)

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

target_link_libraries(vma_raii
    PUBLIC
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
)

target_include_directories(vma_raii
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)