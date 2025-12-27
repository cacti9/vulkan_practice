/**
 * @file   vma_raii.h
 * @brief  Simple RAII wrapper for vma.
 * 
 */
#pragma once
#include "vk_mem_alloc.h"

/**
 * @brief for vma_raii::Allocator::imagesMap
 * 
 * VkImage
 * - is a pointer if it is 64bit
 * - is uint64_t otherwise
 * 
 * So just mush them into uint64_t. It is dependent on Vulkan implementation.
 */
template<> struct std::hash<VkImage> {
  size_t operator()(VkImage const& image) const noexcept {
    return std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(image));
  }
};

/**
 * @brief namespace for vma_raii library.
 */
namespace vma_raii {
  /**
   * @brief used for picking physical device.
   */
  extern std::vector<const char*> vmaAskedExtensions;
  /**
   * @brief used for picking physical device.
   */
  extern std::unordered_map<const char*, VmaAllocatorCreateFlagBits> vmaFlagFromExtension;

  /**
   * @brief Singleton class that manages all resources with vma. Must run init() before using.
   * - Deallocates all buffer in destructor.
   * - StagingBuffer is the only public allocation.
   */
  class Allocator { 
  private:
    /**
     * @brief RAII wrapper for vk::Image
     */
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

    /**
     * @brief RAII wrapper for vk::Buffer
     */
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

    /**
     * @brief unmaps allocation before destroy.
     */
    struct MappedBuffer : Buffer {
      MappedBuffer(VkDeviceSize size) 
        : Buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT) {}
      ~MappedBuffer() override {
        vmaUnmapMemory(vmaAllocator, allocation);
      }
    };

  public:
    /**
     * @brief The only public resource in Allocator.
     */
    class StagingBuffer : protected Buffer { 
    public:
      vk::Buffer getBuffer() { return buffer; }
    private:
      friend class Allocator;
      StagingBuffer(VkDeviceSize size)
        : Buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) {}
    };

    /**
     * @brief RUN FIRST, only after creating logical device.
     */
    void init(VmaAllocatorCreateFlags vmaAvailableFlags, vk::PhysicalDevice physicalDevice, vk::Device device, vk::Instance instance);

    /**
     * @brief creates vk::Image from info and overrides.
     * @param[in,out] image If it is not empty, it is deallocated first before getting new image.
     */
    void createImage(vk::ImageCreateInfo imageInfo, vk::Image& image);

    /**
     * @brief creates raii buffer that deallocates after scope.
     */
    StagingBuffer createStagingBuffer(VkDeviceSize size, const void* data);

    /**
     * @brief Usually for index/vertex buffer.
     * @return A handle for buffer. Buffer will live in Allocator instance.
     */
    vk::Buffer createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage);

    /**
     * @brief Usually for uniform buffer.
     * @return A handle and a mapped pointer.
     */
    std::pair<vk::Buffer, void*> createMappedBuffer(VkDeviceSize size);

    Allocator() = default;
    ~Allocator() {
      buffers.clear();
      imagesMap.clear();
      if (vmaAllocator) {
        vmaDestroyAllocator(vmaAllocator);
      }
    }
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
  private:
    std::vector<std::unique_ptr<Buffer>> buffers; /**< Where allocated buffers live */
    std::unordered_map<VkImage, Image> imagesMap; /**< Images must be searched before creation in createImage()*/
    inline static VmaAllocator vmaAllocator; /**< Singleton */
  }; 
}