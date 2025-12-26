#pragma once
#include "vk_mem_alloc.h"

template<> struct std::hash<VkImage> {
  size_t operator()(VkImage const& image) const noexcept {
    return std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(image));
  }
};

namespace vma_raii {
  extern std::vector<const char*> vmaAskedExtensions;
  extern std::unordered_map<const char*, VmaAllocatorCreateFlagBits> vmaFlagFromExtension;

  class Allocator { 
  private:
    struct Image {
      Image() = default;
      Image(vk::ImageCreateInfo imageInfo);
      virtual ~Image() {
        if (image && allocation) {
          vmaDestroyImage(vmaAllocator, image, allocation);
          image = nullptr;
          allocation = nullptr;
        }
      }
      Image(const Image&) = delete;
      Image& operator=(const Image&) = delete;
      Image(Image&& other) noexcept
        : image(other.image), allocation(other.allocation) {
        other.image = nullptr;
        other.allocation = nullptr;
      }
      Image& operator=(Image&& other) noexcept = delete;

      vk::Image image;
      VmaAllocation allocation;
    };

    struct Buffer {
      Buffer() = default;
      Buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags flags = 0u);
      virtual ~Buffer() {
        if (buffer && allocation) {
          vmaDestroyBuffer(vmaAllocator, buffer, allocation);
          buffer = nullptr;
          allocation = nullptr;
        }
      }
      Buffer(const Buffer&) = delete;
      Buffer& operator=(const Buffer&) = delete;
      Buffer(Buffer&& other) noexcept 
        : buffer(other.buffer), allocation(other.allocation) {
          other.buffer = nullptr;
          other.allocation = nullptr;
      }
      Buffer& operator=(Buffer&& other) noexcept = delete;

      vk::Buffer buffer;
      VmaAllocation allocation;
    };

    struct MappedBuffer : Buffer {
      MappedBuffer(VkDeviceSize size) 
        : Buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT) {}
      ~MappedBuffer() override {
        vmaUnmapMemory(vmaAllocator, allocation);
      }
    };

  public:
    class StagingBuffer : protected Buffer {
    public:
      vk::Buffer getBuffer() { return buffer; }
    private:
      friend class Allocator;
      StagingBuffer(VkDeviceSize size)
        : Buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) {}
    };

    // member functions
    void createImage(vk::ImageCreateInfo imageInfo, vk::Image& image);
    StagingBuffer createStagingBuffer(VkDeviceSize size, const void* pSrcHostPointer);
    vk::Buffer createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
    std::pair<vk::Buffer, void*> createMappedBuffer(VkDeviceSize size);

    Allocator() = default;
    void init(VmaAllocatorCreateFlags vmaAvailableFlags, vk::PhysicalDevice physicalDevice, vk::Device device, vk::Instance instance);
    ~Allocator() {
      buffers.clear();
      imagesMap.clear();
      if (vmaAllocator) {
        vmaDestroyAllocator(vmaAllocator);
      }
    }
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    inline static VmaAllocator vmaAllocator;

  private:
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::unordered_map<VkImage, Image> imagesMap;
  }; 
}