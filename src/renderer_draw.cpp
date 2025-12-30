#include "renderer.h"
#include <chrono>

void VulkanRenderer::drawFrame() {
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
      }
      else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
      }
    }
    catch (const vk::SystemError& e) {
      if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
        recreateSwapChain();
        return;
      }
      else {
        throw;
      }
    }
  }
  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::recreateSwapChain() {
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
void VulkanRenderer::cleanupSwapChain() {
  swapChainImageViews.clear();
  swapChain = nullptr;
}

void VulkanRenderer::updateUniformBuffer(uint32_t currentImage) {
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

void VulkanRenderer::recordCombinedCommandBuffer(uint32_t imageIndex) {
  commandBuffers[currentFrame].reset();
  commandBuffers[currentFrame].begin({});
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
    .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
    .layerCount = 1,
    .colorAttachmentCount = 1,
    .pColorAttachments = &colorAttachmentInfo,
    .pDepthAttachment = &depthAttachmentInfo,
  };
  commandBuffers[currentFrame].beginRendering(originalRenderingInfo);
  commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *originalGraphicsPipeline);
  commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
  commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
  commandBuffers[currentFrame].bindVertexBuffers(0, { vertexBuffer }, { 0 });
  commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
  commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, originalPipelineLayout, 0, *originalDescriptorSets[currentFrame], nullptr);
  commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
  commandBuffers[currentFrame].endRendering();

  vk::RenderingAttachmentInfo attachmentInfo = { 
    .imageView = swapChainImageViews[imageIndex],
    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eLoad,
    .storeOp = vk::AttachmentStoreOp::eStore,
    .clearValue = clearColor
  };
  vk::RenderingInfo renderingInfo = {
    .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
    .layerCount = 1,
    .colorAttachmentCount = 1,
    .pColorAttachments = &attachmentInfo
  };
  commandBuffers[currentFrame].beginRendering(renderingInfo);
  commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
  commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
  commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
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

void VulkanRenderer::recordImageBarrier(
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

void VulkanRenderer::recordComputeCommandBuffer() {
  computeCommandBuffers[currentFrame].reset();
  computeCommandBuffers[currentFrame].begin({});
  computeCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
  computeCommandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, { computeDescriptorSets[currentFrame] }, {});
  computeCommandBuffers[currentFrame].dispatch(PARTICLE_COUNT / 256, 1, 1);
  computeCommandBuffers[currentFrame].end();
}

