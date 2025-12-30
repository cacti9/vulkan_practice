#include <iostream>
#include <vector>
#include <array>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include "vma_raii.h"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t PARTICLE_COUNT = 256;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
  }

  static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
    return {
      vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
      vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),
    };
  }
};
struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
  }

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
    return {
      vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
      vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
      vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
    };
  }

  bool operator==(const Vertex& other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord;
  }
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
  float deltaTime;
};

class VulkanRenderer {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *                     window = nullptr;
  vk::raii::Context                context;
  vk::raii::Instance               instance       = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR             surface        = nullptr;
  vk::raii::PhysicalDevice         physicalDevice = nullptr;

  vk::raii::Device                 device         = nullptr;
  uint32_t                         queueIndex     = ~0;
  vk::raii::Queue                  allInOneQueue  = nullptr;

  vk::raii::SwapchainKHR           swapChain      = nullptr;
  std::vector<vk::Image>           swapChainImages;
  vk::SurfaceFormatKHR             swapChainSurfaceFormat;
  vk::Extent2D                     swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;

  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
  vk::raii::DescriptorSetLayout originalDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout originalPipelineLayout = nullptr;
  vk::raii::Pipeline originalGraphicsPipeline = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
  vk::raii::PipelineLayout computePipelineLayout = nullptr;
  vk::raii::Pipeline computePipeline = nullptr;

  std::vector<vk::Buffer> shaderStorageBuffers;

  vk::Image colorImage;
  vk::raii::ImageView colorImageView = nullptr;
  vk::Image depthImage;
  vk::raii::ImageView depthImageView = nullptr;

  uint32_t mipLevels = 0;
  vk::Image textureImage;
  vk::raii::ImageView textureImageView = nullptr;
  vk::raii::Sampler textureSampler = nullptr;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vk::Buffer vertexBuffer;
  vk::Buffer indexBuffer;

  std::vector<vk::Buffer> uniformBuffers;
  std::vector<void*> uniformBuffersMapped;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> originalDescriptorSets;
  std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;
  std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

  vk::raii::Semaphore semaphore = nullptr;
  uint64_t timelineValue = 0;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t currentFrame = 0;

  bool framebufferResized = false;
  double lastFrameTime = 0.0;
  double lastTime = 0.0f;

  std::vector<const char*> requiredDeviceExtension = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
  };

  VmaAllocatorCreateFlags vmaAvailableFlags = 0;
  vma_raii::Allocator vmaAllocator;

  void initWindow();
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();

    createLogicalDevice();

    createSwapChain();
    createSwapChainImageViews(); 

    createOriginalDescriptorSetLayout();
    createOriginalGraphicsPipeline();
    createComputeDescriptorSetLayout();
    createGraphicsPipeline();
    createComputePipeline();

    // texture mipmap blit requires command buffer
    createCommandPool(); 

    // Resources
    vmaAllocator.init(vmaAvailableFlags, physicalDevice, device, instance);
    createShaderStorageBuffers();
    createColorResources();
    createDepthResources();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer(); // copying from staging requires command buffer
    createIndexBuffer();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets(); // needs sampler & uniform buffer
    createComputeDescriptorSets();

    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
  }

  void mainLoop();
  void cleanup();
  void createInstance();
  [[nodiscard]] std::vector<const char*> getRequiredExtensions() const;
  void setupDebugMessenger();
  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);
  void createSurface();
  void pickPhysicalDevice();
  vk::SampleCountFlagBits getMaxUsableSampleCount();

  void createLogicalDevice();

  void createSwapChain();
  static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const & surfaceCapabilities);
  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
  static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
  void createSwapChainImageViews();

  void createOriginalDescriptorSetLayout();
  void createOriginalGraphicsPipeline();
  void createComputeDescriptorSetLayout();
  void createGraphicsPipeline();
  void createComputePipeline();
  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
  static std::vector<char> readFile(const std::string& filename);

  void createCommandPool();

  void createShaderStorageBuffers();
  void createColorResources();
  void createDepthResources();
  vk::Format findDepthFormat();
  vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const;
  bool hasStencilComponent(vk::Format format);
  void createTextureImage();
  void transitionImageLayout(const vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels);
  void copyBufferToImage(const vk::Buffer& buffer, const vk::Image& image, uint32_t width, uint32_t height);
  void generateMipmaps(const vk::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
  void createTextureImageView();
  void createTextureSampler();
  void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples,
                    vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::Image& image);
  [[nodiscard]] vk::raii::ImageView createImageView(vk::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);
  void loadModel();
  void createVertexBuffer();
  void createIndexBuffer();
  void createDeviceLocalBuffer(vk::Buffer& buffer, VkDeviceSize size, const void* data, VkBufferUsageFlags usage);
  void copyBuffer(const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, VkDeviceSize size);
  vk::raii::CommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void createComputeDescriptorSets();

  void createCommandBuffers();
  void createComputeCommandBuffers();
  void createSyncObjects();

  void drawFrame();
  void recreateSwapChain();
  void cleanupSwapChain();
  void updateUniformBuffer(uint32_t currentImage);
  void recordCombinedCommandBuffer(uint32_t imageIndex);
  void recordImageBarrier(
    vk::Image image,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags image_aspect_flags
  );
  void recordComputeCommandBuffer();
};