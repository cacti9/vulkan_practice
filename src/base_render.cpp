#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>
#include <assert.h>
#include <unordered_map>
#include <random>

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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "vma_raii.h"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr uint32_t PARTICLE_COUNT = 256;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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

template<> struct std::hash<Vertex> {
  size_t operator()(Vertex const& vertex) const noexcept {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    float deltaTime;
};

class HelloTriangleApplication {
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

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();

        createLogicalDevice();

        createSwapChain();
        createSwapChainImageViews(); 

        msaaSamples = getMaxUsableSampleCount();
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

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;
        }

        device.waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{ .pApplicationName   = "Hello Triangle",
                    .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                    .pEngineName        = "No Engine",
                    .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                    .apiVersion         = vk::ApiVersion14 };

        // Get the required layers
        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) {
          requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        for (auto const& requiredLayer : requiredLayers)
        {
            if (std::ranges::none_of(layerProperties,
                                     [requiredLayer](auto const& layerProperty)
                                     { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
            {
                throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
            }
        }

        // Get the required extensions.
        auto requiredExtensions = getRequiredExtensions();

        // Check if the required extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const& requiredExtension : requiredExtensions)
        {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const& extensionProperty)
                                     { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames     = requiredLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data() };
        instance = vk::raii::Instance(context, createInfo);
    }
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
            .messageSeverity = severityFlags,
            .messageType = messageTypeFlags,
            .pfnUserCallback = &debugCallback
            };
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto                            devIter = std::ranges::find_if(
          devices,
          [&]( auto const & device )
          {
            // Check if the device supports the Vulkan 1.3 API version
            bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

            // Check if any of the queue families support graphics operations
            auto queueFamilies = device.getQueueFamilyProperties();
            bool supportsGraphics =
              std::ranges::any_of( queueFamilies, []( auto const & qfp ) { return !!( qfp.queueFlags & vk::QueueFlagBits::eGraphics ); } );

            // Check if all required device extensions are available
            auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
            bool supportsAllRequiredExtensions =
              std::ranges::all_of( requiredDeviceExtension,
                                   [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
                                   {
                                     return std::ranges::any_of( availableDeviceExtensions,
                                                                 [requiredDeviceExtension]( auto const & availableDeviceExtension )
                                                                 { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; } );
                                   } );

            auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, 
                                                         vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                                                         vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy && 
                                            features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
                                            features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;

            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
          } );
        if ( devIter != devices.end() )
        {
            physicalDevice = *devIter;
            auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
            for (auto vmaAskedExtension : vma_raii::vmaAskedExtensions) {
              bool available = std::ranges::any_of(availableDeviceExtensions,
                                             [vmaAskedExtension](auto const & availableDeviceExtension)
                                             { return strcmp(availableDeviceExtension.extensionName, vmaAskedExtension) == 0; });
              if (available) {
                requiredDeviceExtension.push_back(vmaAskedExtension);
                vmaAvailableFlags |= vma_raii::vmaFlagFromExtension[vmaAskedExtension];
              }
            }
        }
        else
        {
            throw std::runtime_error( "failed to find a suitable GPU!" );
        }
    }

    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports both graphics and present
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
            {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0)
        {
            throw std::runtime_error("Could not find a queue for graphics, compute, and present -> terminating");
        }

        // query for required features (Vulkan 1.1 and 1.3)
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, 
                           vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, 
                           vk::PhysicalDeviceMaintenance5Features, vk::PhysicalDeviceBufferDeviceAddressFeatures, 
                           vk::PhysicalDeviceMemoryPriorityFeaturesEXT, vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR
        > featureChain = {
            {.features = {.samplerAnisotropy = true } },            // vk::PhysicalDeviceFeatures2
            { .shaderDrawParameters = true },                       // vk::PhysicalDeviceVulkan11Features
            { .synchronization2 = true, .dynamicRendering = true, 
                .maintenance4 = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT)}, // vk::PhysicalDeviceVulkan13Features
            { .extendedDynamicState = true },                        // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            { .maintenance5 = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT)},
            { .bufferDeviceAddress = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT)},
            { .memoryPriority = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT)},
            { .timelineSemaphore = true }                            // vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR
        };

        // create a Device
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority };
        vk::DeviceCreateInfo      deviceCreateInfo{ .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                                    .queueCreateInfoCount = 1,
                                                    .pQueueCreateInfos = &deviceQueueCreateInfo,
                                                    .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                                                    .ppEnabledExtensionNames = requiredDeviceExtension.data() };

        device = vk::raii::Device( physicalDevice, deviceCreateInfo );
        allInOneQueue = vk::raii::Queue( device, queueIndex, 0 );
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR( *surface );
        swapChainExtent          = chooseSwapExtent( surfaceCapabilities );
        swapChainSurfaceFormat   = chooseSwapSurfaceFormat( physicalDevice.getSurfaceFormatsKHR( *surface ) );
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{ .surface          = *surface,
                                                        .minImageCount    = chooseSwapMinImageCount( surfaceCapabilities ),
                                                        .imageFormat      = swapChainSurfaceFormat.format,
                                                        .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
                                                        .imageExtent      = swapChainExtent,
                                                        .imageArrayLayers = 1,
                                                        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                                                        .imageSharingMode = vk::SharingMode::eExclusive,
                                                        .preTransform     = surfaceCapabilities.currentTransform,
                                                        .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                        .presentMode      = chooseSwapPresentMode( physicalDevice.getSurfacePresentModesKHR( *surface ) ),
                                                        .clipped          = true };

        swapChain = vk::raii::SwapchainKHR( device, swapChainCreateInfo );
        swapChainImages = swapChain.getImages();
    }
    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const & surfaceCapabilities) {
        auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        assert(!availableFormats.empty());
        const auto formatIt = std::ranges::find_if(
            availableFormats,
            []( const auto & format ) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; } );
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }
    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        assert(std::ranges::any_of(availablePresentModes, [](auto presentMode){ return presentMode == vk::PresentModeKHR::eFifo; }));
        return std::ranges::any_of(availablePresentModes,
            [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != 0xFFFFFFFF) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void createSwapChainImageViews() {
        assert(swapChainImageViews.empty());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
          swapChainImageViews.push_back(createImageView(swapChainImages[i], swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
        }
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount() {
      vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

      vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
      if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
      if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
      if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
      if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
      if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
      if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

      return vk::SampleCountFlagBits::e1;
    }

    void createOriginalDescriptorSetLayout() {
      std::array bindings = {
          vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
          vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
      };

      vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() };
      originalDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }
    void createOriginalGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(SHADER_DIR) + "/original.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule,  .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
          .vertexBindingDescriptionCount =1,
          .pVertexBindingDescriptions = &bindingDescription,
          .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
          .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
          .topology = vk::PrimitiveTopology::eTriangleList
        };
        vk::PipelineViewportStateCreateInfo viewportState{
          .viewportCount = 1,
          .scissorCount = 1
        };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
          .depthClampEnable = vk::False,
          .rasterizerDiscardEnable = vk::False,
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .depthBiasEnable = vk::False,
          .depthBiasSlopeFactor = 1.0f,
          .lineWidth = 1.0f
        };
        vk::PipelineMultisampleStateCreateInfo multisampling{
          .rasterizationSamples = msaaSamples,
          .sampleShadingEnable = vk::False
        };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
          .blendEnable = vk::False,
          .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
          .logicOpEnable = vk::False,
          .logicOp =  vk::LogicOp::eCopy,
          .attachmentCount = 1,
          .pAttachments =  &colorBlendAttachment
        };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{  .setLayoutCount = 1, .pSetLayouts = &*originalDescriptorSetLayout, .pushConstantRangeCount = 0 };

        originalPipelineLayout = vk::raii::PipelineLayout( device, pipelineLayoutInfo );

        vk::Format depthFormat = findDepthFormat();

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
          {.stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = originalPipelineLayout,
            .renderPass = nullptr },
          {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format, .depthAttachmentFormat = depthFormat}
        };

        originalGraphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void createComputeDescriptorSetLayout() {
      std::array layoutBindings{
          vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
          vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
          vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
      };

      vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
      computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }


    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(SHADER_DIR) + "/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule,  .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
          .vertexBindingDescriptionCount =1,
          .pVertexBindingDescriptions = &bindingDescription,
          .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
          .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
          .topology = vk::PrimitiveTopology::ePointList
        };
        vk::PipelineViewportStateCreateInfo viewportState{
          .viewportCount = 1,
          .scissorCount = 1
        };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
          .depthClampEnable = vk::False,
          .rasterizerDiscardEnable = vk::False,
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .depthBiasEnable = vk::False,
          .depthBiasSlopeFactor = 1.0f,
          .lineWidth = 1.0f
        };
        vk::PipelineMultisampleStateCreateInfo multisampling{
          .rasterizationSamples = vk::SampleCountFlagBits::e1,
          .sampleShadingEnable = vk::False
        };
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
          .blendEnable = vk::True,
          .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
          .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
          .colorBlendOp = vk::BlendOp::eAdd,
          .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
          .dstAlphaBlendFactor = vk::BlendFactor::eZero,
          .alphaBlendOp = vk::BlendOp::eAdd,
          .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };


        vk::PipelineColorBlendStateCreateInfo colorBlending{
          .logicOpEnable = vk::False,
          .logicOp =  vk::LogicOp::eCopy,
          .attachmentCount = 1,
          .pAttachments =  &colorBlendAttachment
        };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;

        auto pipelineLayout = vk::raii::PipelineLayout( device, pipelineLayoutInfo );

        vk::Format depthFormat = findDepthFormat();

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
          {.stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = nullptr },
          {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format, }
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }
    void createComputePipeline() {
      vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

      vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "compMain" };
      vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*computeDescriptorSetLayout };
      computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
      vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *computePipelineLayout };
      computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }
    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{
          .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          .queueFamilyIndex = queueIndex
        };
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    // <Resources>-----------------------------------------
    void createShaderStorageBuffers() {
      // Initialize particles
      std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
      std::uniform_real_distribution rndDist(0.0f, 1.0f);

      // Initial particle positions on a circle
      std::vector<Particle> particles(PARTICLE_COUNT);
      for (auto& particle : particles) {
        float r = 0.25f * sqrtf(rndDist(rndEngine));
        float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
        float x = r * cosf(theta) * HEIGHT / WIDTH;
        float y = r * sinf(theta);
        particle.position = glm::vec2(x, y);
        particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
        particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
      }

      vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
      auto stagingBuffer = vmaAllocator.createStagingBuffer(bufferSize, particles.data());

      shaderStorageBuffers.clear();
      // Copy initial particle data to all storage buffers
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        auto shaderStorageBufferTemp = vmaAllocator.createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        copyBuffer(stagingBuffer.getBuffer(), shaderStorageBufferTemp, bufferSize);
        shaderStorageBuffers.push_back(shaderStorageBufferTemp);
      }
    }

    void createColorResources() {
      vk::Format colorFormat = swapChainSurfaceFormat.format;

      createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, colorImage);
      colorImageView = createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    void createDepthResources() {
      vk::Format depthFormat = findDepthFormat();

      createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, depthImage);
      depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
    }
    vk::Format findDepthFormat() {
      return findSupportedFormat(
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
      );
    }
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }
    bool hasStencilComponent(vk::Format format) {
      return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void createTextureImage() {
      int texWidth, texHeight, texChannels;
      stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
      vk::DeviceSize imageSize = texWidth * texHeight * 4;
      mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

      if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
      }

      auto stagingBuffer = vmaAllocator.createStagingBuffer(imageSize, pixels);

      stbi_image_free(pixels);

      createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, textureImage);

      transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
      copyBufferToImage(stagingBuffer.getBuffer(), textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

      generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    }
    void transitionImageLayout(const vk::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
      auto commandBuffer = beginSingleTimeCommands();

      vk::ImageMemoryBarrier barrier{
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .image = image,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1 }
      };

      vk::PipelineStageFlags sourceStage;
      vk::PipelineStageFlags destinationStage;

      if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
      }
      else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
      }
      else {
        throw std::invalid_argument("unsupported layout transition!");
      }
      commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
      endSingleTimeCommands(commandBuffer);
    }
    void copyBufferToImage(const vk::Buffer& buffer, const vk::Image& image, uint32_t width, uint32_t height) {
      auto commandBuffer = beginSingleTimeCommands();
      vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1}
      };
      commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
      endSingleTimeCommands(commandBuffer);
    }
    void generateMipmaps(const vk::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
      // Check if image format supports linear blit-ing
      vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

      if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
      }

      vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

      vk::ImageMemoryBarrier barrier = { 
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image
      };
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barrier.subresourceRange.levelCount = 1;

      int32_t mipWidth = texWidth;
      int32_t mipHeight = texHeight;

      for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

        vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
        offsets[0] = vk::Offset3D(0, 0, 0);
        offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
        dstOffsets[0] = vk::Offset3D(0, 0, 0);
        dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);
        vk::ImageBlit blit = { .srcSubresource = {}, .srcOffsets = offsets,
                            .dstSubresource = {}, .dstOffsets = dstOffsets };
        blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
        blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

        commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
      }

      barrier.subresourceRange.baseMipLevel = mipLevels - 1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

      endSingleTimeCommands(commandBuffer);
    }

    void createTextureImageView() {
      textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
    }

    void createTextureSampler() {
      vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
      vk::SamplerCreateInfo samplerInfo{
          .magFilter = vk::Filter::eLinear,
          .minFilter = vk::Filter::eLinear,
          .mipmapMode = vk::SamplerMipmapMode::eLinear,
          .addressModeU = vk::SamplerAddressMode::eRepeat,
          .addressModeV = vk::SamplerAddressMode::eRepeat,
          .addressModeW = vk::SamplerAddressMode::eRepeat,
          .mipLodBias = 0.0f,
          .anisotropyEnable = vk::True,
          .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
          .compareEnable = vk::False,
          .compareOp = vk::CompareOp::eAlways,
          .minLod = 0.0f,
          .maxLod = vk::LodClampNone
      };
      textureSampler = vk::raii::Sampler(device, samplerInfo);
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, 
                     vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::Image& image) {
      vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = numSamples,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive
      };
      vmaAllocator.createImage(imageInfo, image);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(vk::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
      vk::ImageViewCreateInfo viewInfo{
          .image = image,
          .viewType = vk::ImageViewType::e2D,
          .format = format,
          .subresourceRange = { aspectFlags, 0, mipLevels, 0, 1 }
      };
      return vk::raii::ImageView(device, viewInfo);
    }
    // </Resources>----------------------------------------

    void loadModel() {
      tinyobj::attrib_t attrib;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string warn, err;

      if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
        throw std::runtime_error(warn + err);
      }

      std::unordered_map<Vertex, uint32_t> uniqueVertices{};

      for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
          Vertex vertex{};

          vertex.pos = {
              attrib.vertices[3 * index.vertex_index + 0],
              attrib.vertices[3 * index.vertex_index + 1],
              attrib.vertices[3 * index.vertex_index + 2]
          };

          vertex.texCoord = {
              attrib.texcoords[2 * index.texcoord_index + 0],
              1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
          };

          vertex.color = { 1.0f, 1.0f, 1.0f };

          if (!uniqueVertices.contains(vertex)) {
            uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
            vertices.push_back(vertex);
          }

          indices.push_back(uniqueVertices[vertex]);
        }
      }
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        createDeviceLocalBuffer(vertexBuffer, bufferSize, vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        createDeviceLocalBuffer(indexBuffer, bufferSize, indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    void createDeviceLocalBuffer(vk::Buffer& buffer, VkDeviceSize size, const void* data, VkBufferUsageFlags usage) {
        auto stagingBuffer = vmaAllocator.createStagingBuffer(size, data);
        buffer = vmaAllocator.createDeviceLocalBuffer(size, usage);
        copyBuffer(stagingBuffer.getBuffer(), buffer, size);
    }
    void copyBuffer(const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, VkDeviceSize size) {
      auto commandCopyBuffer = beginSingleTimeCommands();
      commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
      endSingleTimeCommands(commandCopyBuffer);
    }

    void createUniformBuffers() {
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        auto [buf, map] = vmaAllocator.createMappedBuffer(sizeof(UniformBufferObject));
        uniformBuffersMapped.push_back(map);
        uniformBuffers.push_back(buf);
      }
    }

    void createDescriptorPool() {
      std::array poolSize{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT*2), //todo
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)
      };
      vk::DescriptorPoolCreateInfo poolInfo{
          .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
          .maxSets = MAX_FRAMES_IN_FLIGHT*2,
          .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
          .pPoolSizes = poolSize.data()
      };
      descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
      std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, originalDescriptorSetLayout);
      vk::DescriptorSetAllocateInfo allocInfo{
          .descriptorPool = descriptorPool,
          .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
          .pSetLayouts = layouts.data()
      };

      originalDescriptorSets.clear();
      originalDescriptorSets = device.allocateDescriptorSets(allocInfo);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = uniformBuffers[i],
            .offset = 0,
            .range = sizeof(UniformBufferObject)
        };
        vk::DescriptorImageInfo imageInfo{
            .sampler = textureSampler,
            .imageView = textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };
        std::array descriptorWrites{
            vk::WriteDescriptorSet{
                .dstSet = originalDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
            },
            vk::WriteDescriptorSet{
                .dstSet = originalDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo
            }
        };
        device.updateDescriptorSets(descriptorWrites, {});
      }
    }
    void createComputeDescriptorSets() {
      std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
      vk::DescriptorSetAllocateInfo allocInfo{};
      allocInfo.descriptorPool = *descriptorPool;
      allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
      allocInfo.pSetLayouts = layouts.data();
      computeDescriptorSets.clear();
      computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

        vk::DescriptorBufferInfo storageBufferInfoLastFrame(shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
        vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shaderStorageBuffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);
        std::array descriptorWrites{
            vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pImageInfo = nullptr, .pBufferInfo = &bufferInfo, .pTexelBufferView = nullptr },
            vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoLastFrame, .pTexelBufferView = nullptr },
            vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoCurrentFrame, .pTexelBufferView = nullptr },
        };
        device.updateDescriptorSets(descriptorWrites, {});
      }
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{
          .commandPool = commandPool,
          .level = vk::CommandBufferLevel::ePrimary,
          .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };
        commandBuffers = vk::raii::CommandBuffers( device, allocInfo );
    }
    void createComputeCommandBuffers() {
      computeCommandBuffers.clear();
      vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
      };
      computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void createSyncObjects() {
      inFlightFences.clear();

      vk::SemaphoreTypeCreateInfo semaphoreType{ .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0 };
      semaphore = vk::raii::Semaphore(device, { .pNext = &semaphoreType });
      timelineValue = 0;

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::FenceCreateInfo fenceInfo{};
        inFlightFences.emplace_back(device, fenceInfo);
      }
    }

    void drawFrame() {
      auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, nullptr, *inFlightFences[currentFrame]);
      while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
        ;
      device.resetFences(*inFlightFences[currentFrame]);

      // Update timeline value for this frame
      uint64_t computeWaitValue = timelineValue;
      uint64_t computeSignalValue = ++timelineValue;
      uint64_t graphicsWaitValue = computeSignalValue;
      uint64_t graphicsSignalValue = ++timelineValue;

      updateUniformBuffer(currentFrame);

      {
        recordComputeCommandBuffer();
        vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
          .waitSemaphoreValueCount = 1,
          .pWaitSemaphoreValues = &computeWaitValue,
          .signalSemaphoreValueCount = 1,
          .pSignalSemaphoreValues = &computeSignalValue
        };
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;

        vk::SubmitInfo computeSubmitInfo{
          .pNext = &computeTimelineInfo,
          .waitSemaphoreCount = 1,
          .pWaitSemaphores = &*semaphore,
          .pWaitDstStageMask = &waitStage,
          .commandBufferCount = 1,
          .pCommandBuffers = &*computeCommandBuffers[currentFrame],
          .signalSemaphoreCount = 1,
          .pSignalSemaphores = &*semaphore
        };

        allInOneQueue.submit(computeSubmitInfo, nullptr);
      }
      {
        recordCombinedCommandBuffer(imageIndex);

        vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
          .waitSemaphoreValueCount = 1,
          .pWaitSemaphoreValues = &graphicsWaitValue,
          .signalSemaphoreValueCount = 1,
          .pSignalSemaphoreValues = &graphicsSignalValue
        };
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;

        vk::SubmitInfo graphicsSubmitInfo{
          .pNext = &graphicsTimelineInfo,
          .waitSemaphoreCount = 1,
          .pWaitSemaphores = &*semaphore,
          .pWaitDstStageMask = &waitStage,
          .commandBufferCount = 1,
          .pCommandBuffers = &*commandBuffers[currentFrame],
          .signalSemaphoreCount = 1,
          .pSignalSemaphores = &*semaphore
        };

        allInOneQueue.submit(graphicsSubmitInfo, nullptr);

        vk::SemaphoreWaitInfo waitInfo{
          .semaphoreCount = 1,
          .pSemaphores = &*semaphore,
          .pValues = &graphicsSignalValue
        };

        while (vk::Result::eTimeout == device.waitSemaphores(waitInfo, UINT64_MAX))
          ;

        vk::PresentInfoKHR presentInfo{
          .waitSemaphoreCount = 0,
          .pWaitSemaphores = nullptr,
          .swapchainCount = 1,
          .pSwapchains = &*swapChain,
          .pImageIndices = &imageIndex
        };

        try {
          result = allInOneQueue.presentKHR(presentInfo);
          if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
          } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
          }
        } catch (const vk::SystemError& e) {
          if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
            recreateSwapChain();
            return;
          } else {
            throw;
          }
        }
      }
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createSwapChainImageViews();
        createDepthResources();
        createColorResources();
    }
    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }
    void updateUniformBuffer(uint32_t currentImage) {
      static auto startTime = std::chrono::high_resolution_clock::now();

      auto currentTime = std::chrono::high_resolution_clock::now();
      float time = std::chrono::duration<float>(currentTime - startTime).count() / 3.0f;

      UniformBufferObject ubo{};
      ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
      ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
      ubo.proj[1][1] *= -1;
      ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

      memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
    void recordCombinedCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame].reset();
        commandBuffers[currentFrame].begin( {} );
        // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        recordImageBarrier(
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},                                                     // srcAccessMask (no need to wait for previous operations)
            vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
            vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // dstStage
            vk::ImageAspectFlagBits::eColor
        );
        // Transition the multisampled color image to COLOR_ATTACHMENT_OPTIMAL 
        recordImageBarrier(
          colorImage,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eColorAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          vk::ImageAspectFlagBits::eColor
        );
        // Transition the depth image to DEPTH_ATTACHMENT_OPTIMAL
        recordImageBarrier(
          depthImage,
          vk::ImageLayout::eUndefined,
          vk::ImageLayout::eDepthAttachmentOptimal,
          {},
          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          vk::PipelineStageFlagBits2::eTopOfPipe,
          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
          vk::ImageAspectFlagBits::eDepth
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

        // Color attachment (multisampled) with resolve attachment
        vk::RenderingAttachmentInfo colorAttachmentInfo = {
            .imageView = colorImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eAverage,
            .resolveImageView = swapChainImageViews[imageIndex],
            .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor
        };

        vk::RenderingAttachmentInfo depthAttachmentInfo = {
            .imageView = depthImageView,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clearDepth
        };

        vk::RenderingInfo originalRenderingInfo = {
            .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentInfo,
            .pDepthAttachment = &depthAttachmentInfo,
        };

        vk::RenderingAttachmentInfo attachmentInfo = {
          .imageView = swapChainImageViews[imageIndex],
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eLoad,
          .storeOp = vk::AttachmentStoreOp::eStore,
          .clearValue = clearColor
        };
        vk::RenderingInfo renderingInfo = {
            .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments=&attachmentInfo
        };
        commandBuffers[currentFrame].beginRendering(originalRenderingInfo);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *originalGraphicsPipeline);
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor( 0, vk::Rect2D( vk::Offset2D( 0, 0 ), swapChainExtent ) );
        commandBuffers[currentFrame].bindVertexBuffers(0, { vertexBuffer }, { 0 });
        commandBuffers[currentFrame].bindIndexBuffer( indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value );
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, originalPipelineLayout, 0, *originalDescriptorSets[currentFrame], nullptr);
        commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffers[currentFrame].endRendering();

        commandBuffers[currentFrame].beginRendering(renderingInfo);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor( 0, vk::Rect2D( vk::Offset2D( 0, 0 ), swapChainExtent ) );
        commandBuffers[currentFrame].bindVertexBuffers(0, { shaderStorageBuffers[currentFrame] }, { 0 });
        commandBuffers[currentFrame].draw(PARTICLE_COUNT, 1, 0, 0);
        commandBuffers[currentFrame].endRendering();
        // After rendering, transition the swapchain image to PRESENT_SRC
        recordImageBarrier(
            swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
            {},                                                      // dstAccessMask
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
            vk::PipelineStageFlagBits2::eBottomOfPipe,                  // dstStage
            vk::ImageAspectFlagBits::eColor
        );
        commandBuffers[currentFrame].end();
    }
    void recordImageBarrier(
        vk::Image image,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask,
        vk::ImageAspectFlags image_aspect_flags
        ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = image_aspect_flags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }
    void recordComputeCommandBuffer() {
      computeCommandBuffers[currentFrame].reset();
      computeCommandBuffers[currentFrame].begin({});
      computeCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
      computeCommandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, { computeDescriptorSets[currentFrame] }, {});
      computeCommandBuffers[currentFrame].dispatch(PARTICLE_COUNT / 256, 1, 1);
      computeCommandBuffers[currentFrame].end();
    }

    // used across many functions
    vk::raii::CommandBuffer beginSingleTimeCommands() {
      vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
      };
      vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

      vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
      };
      commandBuffer.begin(beginInfo);

      return commandBuffer;
    }
    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
      commandBuffer.end();

      vk::SubmitInfo submitInfo{
        .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer
      };
      allInOneQueue.submit(submitInfo, nullptr);
      allInOneQueue.waitIdle();
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
