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

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

constexpr uint32_t PERFECT_2SPOWER_RESOLUTION = 1024;
constexpr uint32_t WIDTH = PERFECT_2SPOWER_RESOLUTION;
constexpr uint32_t HEIGHT = PERFECT_2SPOWER_RESOLUTION;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) )
        };
    }
};

//------------------------------------------------------------------------------------------------------------------------------------------
// <From Shader>
// changed Fix definition

#define L 8          // number of 32-bit limbs (example: 256-bit)
#define FRAC_LIMBS 7 // N = 32*7= 224 fractional bits

struct BigUINT {
  uint32_t limb[L]; // little-endian
};
bool isZero(BigUINT a) {
  for (int i = 0; i < L; i++)
    if (a.limb[i] != 0) return false;
  return true;
}
BigUINT operator+(BigUINT a, BigUINT b) {
  BigUINT r;
  uint32_t carry = 0;
  for (int i = 0; i < L; i++)
  {
    uint64_t s = (uint64_t)a.limb[i] + (uint64_t)b.limb[i] + (uint64_t)carry;
    r.limb[i] = (uint32_t)s;
    carry = (uint32_t)(s >> 32);
  }
  return r;
}
// must be a>=b
BigUINT operator-(BigUINT a, BigUINT b) {
  BigUINT r;
  uint32_t borrow = 0;

  for (int i = 0; i < L; i++)
  {
    uint32_t ai = a.limb[i];
    uint32_t bi = b.limb[i];

    // subtract bi + borrow from ai
    uint32_t t = bi + borrow;

    // detect borrow: if ai < t, we need to borrow from next limb
    uint32_t newBorrow = (ai < t) ? 1u : 0u;

    r.limb[i] = ai - t;
    borrow = newBorrow;
  }
  return r;
}
BigUINT shiftLeft(BigUINT a) {
  BigUINT r;
  uint32_t carry = 0;

  for (int i = 0; i < L; i++)
  {
    uint32_t newCarry = (a.limb[i] >> 31) & 1u;
    r.limb[i] = (a.limb[i] << 1) | carry;
    carry = newCarry;
  }

  return r;
}
BigUINT operator>>(BigUINT a, int N) {
  BigUINT r;

  uint32_t wordShift = N >> 5;   // /32
  uint32_t bitShift = N & 31u;  // %32

  // default to zero
  for (int i = 0; i < L; ++i) r.limb[i] = 0u;

  // If shifting out everything -> zero
  if (wordShift >= uint32_t(L)) return r;

  // For each destination limb i, read from source limb (i + wordShift)
  // because limb[0] is least significant.
  for (int i = 0; i < L; ++i)
  {
    int src = i + int(wordShift);
    if (src >= L) break;

    uint32_t lo = a.limb[src];

    if (bitShift == 0u)
    {
      r.limb[i] = lo;
    }
    else
    {
      // bring in high bits from next limb (more significant limb)
      uint32_t hi = (src + 1 < L) ? a.limb[src + 1] : 0u;
      r.limb[i] = (lo >> bitShift) | (hi << (32u - bitShift));
    }
  }

  return r;
}
int compare(BigUINT a, BigUINT b) {
  for (int i = L - 1; i >= 0; i--)
  {
    uint32_t ai = a.limb[i];
    uint32_t bi = b.limb[i];
    if (ai > bi) return 1;
    if (ai < bi) return -1;
  }
  return 0;
}

struct BigUINT2 {
  uint32_t limb[2 * L];
};
BigUINT2 mulWide(BigUINT a, BigUINT b) {
  BigUINT2 p;
  for (int i = 0; i < 2 * L; i++) p.limb[i] = 0;

  for (int i = 0; i < L; i++)
  {
    uint64_t carry = 0;
    for (int j = 0; j < L; j++)
    {
      int k = i + j;
      uint64_t cur = (uint64_t)p.limb[k]
        + (uint64_t)a.limb[i] * (uint64_t)b.limb[j]
        + carry;
      p.limb[k] = (uint32_t)cur;
      carry = cur >> 32;
    }

    // propagate carry across higher limbs
    int k = i + L;
    while ((bool)carry && k < 2 * L)
    {
      uint64_t cur2 = (uint64_t)p.limb[k] + carry;
      p.limb[k] = (uint32_t)cur2;
      carry = cur2 >> 32;
      k++;
    }
  }
  return p;
}
BigUINT2 squareWide(BigUINT a) {
  BigUINT2 p;
  for (int i = 0; i < 2 * L; i++) p.limb[i] = 0;

  for (int i = 0; i < L; i++)
  {
    uint64_t carry = 0;
    for (int j = 0; j < L; j++)
    {
      int k = i + j;
      uint64_t cur = (uint64_t)p.limb[k]
        + (uint64_t)a.limb[i] * (uint64_t)a.limb[j]
        + carry;
      p.limb[k] = (uint32_t)cur;
      carry = cur >> 32;
    }

    // propagate carry across higher limbs
    int k = i + L;
    while ((bool)carry && k < 2 * L)
    {
      uint64_t cur2 = (uint64_t)p.limb[k] + carry;
      p.limb[k] = (uint32_t)cur2;
      carry = cur2 >> 32;
      k++;
    }
  }
  return p;
}

struct Fix {
  BigUINT mag;
  int neg = false;

};
Fix negate(Fix x) {
  x.neg = !x.neg;
  return x;
}
Fix normalizeFix(Fix x) {
  if (isZero(x.mag)) x.neg = false;
  return x;
}
Fix operator+(Fix x, Fix y) {
  Fix r;
  if (x.neg == y.neg)
  {
    r.neg = x.neg;
    r.mag = x.mag + y.mag;
  }
  else {
    int c = compare(x.mag, y.mag);
    if (c == 0) {
      return Fix{0};
    }
    else if (c > 0) {
      r.neg = x.neg;
      r.mag = x.mag - y.mag;
    }
    else {
      r.neg = y.neg;
      r.mag = y.mag - x.mag;
    }
  }
  return r;
}
Fix operator-(Fix x, Fix y) {
  return x + negate(y);
}
Fix shiftLeft(Fix x) {
  Fix y;
  y.neg = x.neg;
  y.mag = shiftLeft(x.mag);
  return normalizeFix(y);
}
Fix operator>>(Fix x, int N) {
  Fix y;
  y.neg = x.neg;
  y.mag = x.mag >> N;
  return normalizeFix(y);
}
Fix FixFromFloat(float x) {
  Fix r{0};
  r.neg = (x < 0.0f);
  float ax = abs(x);

  float ip = floor(ax); //integer part
  r.mag.limb[L - 1] = (uint32_t)ip;
  float frac = ax - ip;

  for (int i = L - 2; i >= 0; --i)
  {
    frac = frac * 4294967296.0;      // 2^32
    float di = floor(frac);
    r.mag.limb[i] = (uint32_t)di;
    frac = frac - di;
  }

  return normalizeFix(r);
}
Fix FixFromDouble(double x) {
  Fix r{0};
  r.neg = (x < 0.0f);
  double ax = abs(x);

  double ip = floor(ax); //integer part
  r.mag.limb[L - 1] = (uint32_t)ip;
  double frac = ax - ip;

  for (int i = L - 2; i >= 0; --i)
  {
    frac = frac * 4294967296.0;      // 2^32
    double di = floor(frac);
    r.mag.limb[i] = (uint32_t)di;
    frac = frac - di;
  }

  return normalizeFix(r);
}
Fix operator*(Fix x, Fix y) {
  Fix r;
  BigUINT2 p = mulWide(x.mag, y.mag);
  for (int i = 0; i < L; i++)
  {
    int src = i + FRAC_LIMBS;
    r.mag.limb[i] = p.limb[src];
  }
  r.neg = x.neg != y.neg;
  return r;
}
Fix square(Fix x) {
  Fix r;
  BigUINT2 p = squareWide(x.mag);
  for (int i = 0; i < L; i++)
  {
    int src = i + FRAC_LIMBS;
    r.mag.limb[i] = p.limb[src];
  }
  r.neg = false;
  return r;
}

struct FixC {
  Fix x;
  Fix y;
};
FixC operator+(FixC z1, FixC z2) {
  FixC r;
  r.x = z1.x + z2.x;
  r.y = z1.y + z2.y;
  return r;
}
// square of complex number z=x+yi
FixC square(FixC z) {
  FixC r;
  r.x = square(z.x) - square(z.y);
  r.y = shiftLeft(z.x * z.y);
  return r;
}
uint32_t length(FixC z) {
  BigUINT r = square(z.x).mag + square(z.y).mag;
  return r.limb[L - 1];
}
//------------------------------------------------------------------------------------------------------------------------------------------
struct alignas(16) UniformBufferObject {
  glm::dvec2 mousePos;
  alignas(16) Fix centerX;
  alignas(16) Fix centerY;
  int shiftN; // 10 - log2(span)
  int iteration_count;
  float     time;
};
UniformBufferObject ubo = {
.centerX{FixFromFloat(0.f)},
.centerY{FixFromFloat(0.f)},
.shiftN{8},
.iteration_count{32},
};
void displayUBO() {
  printf("shiftN: %d\n", ubo.shiftN);
  printf("size of number range: %e\n", glm::exp2(10 - ubo.shiftN));
  printf("iteration: %d\n", ubo.iteration_count);
  printf("\n");
}

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 3, 1, 0
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
    vk::raii::Queue                  queue          = nullptr;
    vk::raii::SwapchainKHR           swapChain      = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    VkBuffer vertexBuffer;
    VmaAllocation vertexAllocation;
    VkBuffer indexBuffer;
    VmaAllocation indexAllocation;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VmaAllocation> uniformBuffersAllocation;
    std::vector<void*> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t semaphoreIndex = 0;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };
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
    VmaAllocatorCreateFlags vmaAvailableFlags = 0;
    VmaAllocator vmaAllocator;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
    static void keyCallback(GLFWwindow* window,
                            int key,
                            int scancode,
                            int action,
                            int mods)
    {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        if (key == GLFW_KEY_W && action == GLFW_PRESS) {
            //printf("%lf %lf\n",ubo.mousePos.x, ubo.mousePos.y);
            Fix mouseX = FixFromDouble(ubo.mousePos.x-512);
            ubo.centerX = ubo.centerX + (mouseX >> ubo.shiftN);
            Fix mouseY = FixFromDouble(ubo.mousePos.y-512);
            ubo.centerY = ubo.centerY + (mouseY >> ubo.shiftN);
            ubo.shiftN += 1;
            displayUBO();
            app->drawFrame();
        }
        if (key == GLFW_KEY_S && action == GLFW_PRESS) {
            ubo.shiftN -= 1;
            displayUBO();
            app->drawFrame();
        }
        if (key == GLFW_KEY_R && action == GLFW_PRESS) {
            ubo.centerX = FixFromFloat(0.f) ,
            ubo.centerY = FixFromFloat(0.f) ,
            ubo.shiftN = 8;
            ubo.iteration_count = 32;
            displayUBO();
            app->drawFrame();
        }
        if (key == GLFW_KEY_A && action == GLFW_PRESS) {
            ubo.iteration_count *= 1.2;
            displayUBO();
            app->drawFrame();
        }
        if (key == GLFW_KEY_D && action == GLFW_PRESS) {
            if (ubo.iteration_count > 32) {
                ubo.iteration_count /= 2;
            }
            displayUBO();
            app->drawFrame();
        }
    }
    static void mouseCallback(GLFWwindow* window, double x, double y) {
      ubo.mousePos = { x,y };
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createVmaAllocator();
        createSwapChain();
        createImageViews();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
      drawFrame();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            //drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void cleanup() {
        vmaDestroyBuffer(vmaAllocator, vertexBuffer, vertexAllocation);
        vmaDestroyBuffer(vmaAllocator, indexBuffer, indexAllocation);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          vmaUnmapMemory(vmaAllocator, uniformBuffersAllocation[i]);
        }
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          vmaDestroyBuffer(vmaAllocator, uniformBuffers[i], uniformBuffersAllocation[i]);
        }
        vmaDestroyAllocator(vmaAllocator);
        glfwDestroyWindow(window);

        glfwTerminate();
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
        createImageViews();
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

            auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
          } );
        if ( devIter != devices.end() )
        {
            physicalDevice = *devIter;
            auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
            for (auto vmaAskedExtension : vmaAskedExtensions) {
              bool available = std::ranges::any_of(availableDeviceExtensions,
                                             [vmaAskedExtension](auto const & availableDeviceExtension)
                                             { return strcmp(availableDeviceExtension.extensionName, vmaAskedExtension) == 0; });
              if (available) {
                requiredDeviceExtension.push_back(vmaAskedExtension);
                vmaAvailableFlags |= vmaFlagFromExtension[vmaAskedExtension];
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
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
            {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        // query for required features (Vulkan 1.1 and 1.3)
        vk::StructureChain<
          vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
          vk::PhysicalDeviceMaintenance5Features, vk::PhysicalDeviceBufferDeviceAddressFeatures, vk::PhysicalDeviceMemoryPriorityFeaturesEXT
        > featureChain = {
            { .features = vk::PhysicalDeviceFeatures{.shaderFloat64 = true, .shaderInt64 = true,}},                                                     // vk::PhysicalDeviceFeatures2
            { .shaderDrawParameters = true },                       // vk::PhysicalDeviceVulkan11Features
            { .synchronization2 = true, .dynamicRendering = true, 
                .maintenance4 = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT)}, // vk::PhysicalDeviceVulkan13Features
            { .extendedDynamicState = true },                        // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            { .maintenance5 = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT)},
            { .bufferDeviceAddress = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT)},
            { .memoryPriority = static_cast<bool>(vmaAvailableFlags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT)}
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
        queue = vk::raii::Queue( device, queueIndex, 0 );
    }

    void createVmaAllocator() {
      VmaAllocatorCreateInfo allocatorCreateInfo = {
          .flags = vmaAvailableFlags,
          .physicalDevice = *physicalDevice,
          .device = *device,
          .instance = *instance,
          .vulkanApiVersion = VK_API_VERSION_1_4,
      };

      vmaCreateAllocator(&allocatorCreateInfo, &vmaAllocator);
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

    void createImageViews() {
        assert(swapChainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo{ .viewType = vk::ImageViewType::e2D, .format = swapChainSurfaceFormat.format,
          .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
        for ( auto image : swapChainImages )
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back( device, imageViewCreateInfo );
        }
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr);
        vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = 1, .pBindings = &uboLayoutBinding };
        descriptorSetLayout = vk::raii::DescriptorSetLayout( device, layoutInfo );
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(SHADER_DIR) + "/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule,  .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo {  .vertexBindingDescriptionCount =1, .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data() };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{  .topology = vk::PrimitiveTopology::eTriangleList };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{  .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False,
                                                              .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone,
                                                              .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False,
                                                              .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp =  vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments =  &colorBlendAttachment };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{  .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0 };

        pipelineLayout = vk::raii::PipelineLayout( device, pipelineLayoutInfo );

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
          {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format }
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{  .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                             .queueFamilyIndex = queueIndex };
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        createDeviceLocalBuffer(vertexBuffer, vertexAllocation, bufferSize, vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        createDeviceLocalBuffer(indexBuffer, indexAllocation, bufferSize, indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   }

    void createUniformBuffers() {
      uniformBuffers.clear();
      uniformBuffersAllocation.clear();

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkBufferCreateInfo bufferInfo = {
          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          .size = sizeof(UniformBufferObject),
          .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        VmaAllocationCreateInfo allocInfo = {
          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
          .usage = VMA_MEMORY_USAGE_AUTO
        };

        VkBuffer buffer;
        VmaAllocation allocation;
        vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

        void* mapped;
        vmaMapMemory(vmaAllocator, allocation, &mapped);
        uniformBuffersMapped.push_back(mapped);
        uniformBuffers.push_back(buffer);
        uniformBuffersAllocation.push_back(allocation);
      }
    }

    void createDescriptorPool() {
      vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT);
      vk::DescriptorPoolCreateInfo poolInfo{ .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = 1, .pPoolSizes = &poolSize };
      descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
      std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
      vk::DescriptorSetAllocateInfo allocInfo{ .descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data() };

      descriptorSets = device.allocateDescriptorSets(allocInfo);

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{ .buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject) };
        vk::WriteDescriptorSet descriptorWrite{ .dstSet = descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo };
        device.updateDescriptorSets(descriptorWrite, {});
      }
    }

    void createDeviceLocalBuffer(VkBuffer& buffer, VmaAllocation& allocation, VkDeviceSize size, const void* data, VkBufferUsageFlags usage) {
        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;
        VkBufferCreateInfo stagingBufferInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        };
        VmaAllocationCreateInfo stagingAllocInfo = {
          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          .usage = VMA_MEMORY_USAGE_AUTO
        };
        vmaCreateBuffer(vmaAllocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);

        vmaCopyMemoryToAllocation(vmaAllocator, data, stagingAllocation, 0, size);

        VkBufferCreateInfo bufferInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
        };
        VmaAllocationCreateInfo allocInfo = {
          .usage = VMA_MEMORY_USAGE_AUTO
        };
        vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

        copyBuffer(stagingBuffer, buffer, size);
        vmaDestroyBuffer(vmaAllocator, stagingBuffer, stagingAllocation);
    }

    void copyBuffer(VkBuffer & srcBuffer, VkBuffer & dstBuffer, VkDeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
        vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin(vk::CommandBufferBeginInfo { .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        commandCopyBuffer.end();
        queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
        queue.waitIdle();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
                                                 .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
        commandBuffers = vk::raii::CommandBuffers( device, allocInfo );
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame].begin( {} );
        // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},                                                     // srcAccessMask (no need to wait for previous operations)
            vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
            vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
            vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
        );
        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView = swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor
        };
        vk::RenderingInfo renderingInfo = {
            .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo
        };
        commandBuffers[currentFrame].beginRendering(renderingInfo);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor( 0, vk::Rect2D( vk::Offset2D( 0, 0 ), swapChainExtent ) );
        commandBuffers[currentFrame].bindVertexBuffers(0, { vertexBuffer }, { 0 });
        commandBuffers[currentFrame].bindIndexBuffer( indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value );
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
        commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffers[currentFrame].endRendering();
        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
            {},                                                      // dstAccessMask
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
            vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
        );
        commandBuffers[currentFrame].end();
    }

    void transition_image_layout(
        uint32_t imageIndex,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask
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
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
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

    void createSyncObjects() {
        presentCompleteSemaphore.clear();
        renderFinishedSemaphore.clear();
        inFlightFences.clear();

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
             renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
        }


        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            inFlightFences.emplace_back(device, vk::FenceCreateInfo { .flags = vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
      static auto startTime = std::chrono::high_resolution_clock::now();

      auto currentTime = std::chrono::high_resolution_clock::now();
      float time = std::chrono::duration<float>(currentTime - startTime).count();

      ubo.time = time;

      memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        while ( vk::Result::eTimeout == device.waitForFences( *inFlightFences[currentFrame], vk::True, UINT64_MAX ) )
            ;
        auto [result, imageIndex] = swapChain.acquireNextImage( UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr );

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        updateUniformBuffer(currentFrame);

        device.resetFences(  *inFlightFences[currentFrame] );
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );
        const vk::SubmitInfo submitInfo{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*presentCompleteSemaphore[semaphoreIndex],
                            .pWaitDstStageMask = &waitDestinationStageMask, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers[currentFrame],
                            .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore[imageIndex] };
        queue.submit(submitInfo, *inFlightFences[currentFrame]);


        try {
            const vk::PresentInfoKHR presentInfoKHR{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore[imageIndex],
                                                    .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex };
            result = queue.presentKHR( presentInfoKHR );
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
        semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
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

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
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
  printf("Mandelbrot Set Renderer, using 32byte Fixed Points\n");
  printf("W: zoom 2x into mouse position\n");
  printf("S: zoomout 2x\n");
  printf("A: more iteration for divergence check\n");
  printf("D: less iteration\n");
  printf("R: reset variables\n");
  printf("\n");
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
