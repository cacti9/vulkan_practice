#include <vector>
#include <memory>
#include <unordered_map>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define VMA_IMPLEMENTATION
#include "vma_raii.h"

namespace vma_raii { // Start of namespace
  std::vector<const char*> vmaAskedExtensions = {
      vk::KHRDedicatedAllocationExtensionName,
      vk::KHRBindMemory2ExtensionName,
      vk::KHRMaintenance4ExtensionName,
      vk::KHRMaintenance5ExtensionName,
      vk::EXTMemoryBudgetExtensionName,
      vk::KHRBufferDeviceAddressExtensionName,
      vk::EXTMemoryPriorityExtensionName,
      vk::AMDDeviceCoherentMemoryExtensionName,
#ifdef VK_USE_PLATFORM_WIN32_KHR
            vk::KHRExternalMemoryWin32ExtensionName,
#endif
  };
  std::unordered_map<const char*, VmaAllocatorCreateFlagBits> vmaFlagFromExtension = {
    {vk::KHRDedicatedAllocationExtensionName,VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT},
    {vk::KHRBindMemory2ExtensionName, VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT},
    {vk::KHRMaintenance4ExtensionName, VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT},
    {vk::KHRMaintenance5ExtensionName, VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT},
    {vk::EXTMemoryBudgetExtensionName, VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT},
    {vk::KHRBufferDeviceAddressExtensionName, VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT},
    {vk::EXTMemoryPriorityExtensionName, VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT},
    {vk::AMDDeviceCoherentMemoryExtensionName, VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT},
#ifdef VK_USE_PLATFORM_WIN32_KHR
          {vk::KHRExternalMemoryWin32ExtensionName, VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT},
#endif
  };

  void Allocator::init(VmaAllocatorCreateFlags vmaAvailableFlags, vk::PhysicalDevice physicalDevice, vk::Device device, vk::Instance instance) {
    VmaAllocatorCreateInfo allocatorCreateInfo = {
    .flags = vmaAvailableFlags,
    .physicalDevice = physicalDevice,
    .device = device,
    .instance = instance,
    .vulkanApiVersion = VK_API_VERSION_1_4,
    };

    vmaCreateAllocator(&allocatorCreateInfo, &vmaAllocator);
  }

  Allocator::Image::Image(vk::ImageCreateInfo imageInfo)
  {
    VmaAllocationCreateInfo allocInfo = {
      .usage = VMA_MEMORY_USAGE_AUTO
    };
    VkImage temp;
    vmaCreateImage(vmaAllocator, imageInfo, &allocInfo, &temp, &allocation, nullptr);
    image = temp;
  }

  Allocator::Buffer::Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags flags) {
    VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
    };
    VmaAllocationCreateInfo allocInfo = {
      .flags = flags,
      .usage = VMA_MEMORY_USAGE_AUTO
    };
    VkBuffer temp;
    vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocInfo, &temp, &allocation, nullptr);
    buffer = temp;
  }


  void Allocator::createImage(vk::ImageCreateInfo imageInfo, vk::Image& image) {
    imagesMap.erase(image);
    Image raiiImg(imageInfo);
    image = raiiImg.image;
    imagesMap.emplace(image, std::move(raiiImg));
  }
  Allocator::StagingBuffer Allocator::createStagingBuffer(VkDeviceSize size, const void* pSrcHostPointer) {
    StagingBuffer ret(size);
    vmaCopyMemoryToAllocation(vmaAllocator, pSrcHostPointer, ret.allocation, 0, size);
    return ret;
  }
  vk::Buffer Allocator::createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    buffers.push_back(std::make_unique<Buffer>(size, usage));
    return buffers.back()->buffer;
  }
  std::pair<vk::Buffer, void*> Allocator::createMappedBuffer(VkDeviceSize size) {
    buffers.push_back(std::make_unique<MappedBuffer>(size));
    auto& back = buffers.back();
    void* mapped;
    vmaMapMemory(vmaAllocator, back->allocation, &mapped);
    return { back->buffer, mapped };
  }
}
