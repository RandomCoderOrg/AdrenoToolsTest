#include "panvk_present.h"

#include <android/log.h>
#include <dirent.h>
#include <dlfcn.h>
#include <jni.h>
#include <strings.h>
#include <unistd.h>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "panvk_async_spv.h"
#include "panvk_compute_spv.h"
#include "panvk_draw_frag_spv.h"
#include "panvk_draw_vert_spv.h"
#include "panvk_sample_frag_spv.h"
#include "panvk_sample_vert_spv.h"
#include "panvk_triangle_frag_spv.h"
#include "panvk_triangle_vert_spv.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PanvkPresent", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PanvkPresent", __VA_ARGS__)

namespace {
struct ScenePushConstants {
   float monotonic_seconds;
   float width;
   float height;
   float aspect;
};
static_assert(sizeof(ScenePushConstants) == 4 * sizeof(float),
              "scene push constant must be exactly vec4");

struct State {
   void *loader{};
   VkInstance instance{};
   VkPhysicalDevice physical{};
   VkDevice device{};
   VkSurfaceKHR surface{};
   VkSwapchainKHR swapchain{};
   VkRenderPass render_pass{};
   VkPipelineLayout pipeline_layout{};
   VkPipeline pipeline{};
   VkQueue queue{};
   VkCommandPool pool{};
   VkCommandBuffer command{};
   VkSemaphore acquired{};
   VkSemaphore rendered{};
   VkFence fence{};
   uint32_t queue_family{};
   uint32_t frame{};
   VkExtent2D extent{};
   std::string device_name;
   std::chrono::steady_clock::time_point scene_start{};
   std::chrono::steady_clock::time_point stats_window_start{};
   std::chrono::steady_clock::time_point last_successful_present{};
   uint32_t stats_present_calls{};
   uint32_t stats_interval_count{};
   double stats_interval_sum_ms{};
   bool has_successful_present{};
   std::vector<VkImage> images;
   std::vector<VkImageView> image_views;
   std::vector<VkFramebuffer> framebuffers;

   PFN_vkGetInstanceProcAddr GetInstanceProcAddr{};
   PFN_vkGetDeviceProcAddr GetDeviceProcAddr{};
   PFN_vkDestroyInstance DestroyInstance{};
   PFN_vkDestroySurfaceKHR DestroySurfaceKHR{};
   PFN_vkDestroyDevice DestroyDevice{};
   PFN_vkDestroySwapchainKHR DestroySwapchainKHR{};
   PFN_vkDestroyRenderPass DestroyRenderPass{};
   PFN_vkDestroyPipelineLayout DestroyPipelineLayout{};
   PFN_vkDestroyPipeline DestroyPipeline{};
   PFN_vkDestroyImageView DestroyImageView{};
   PFN_vkDestroyFramebuffer DestroyFramebuffer{};
   PFN_vkDestroyCommandPool DestroyCommandPool{};
   PFN_vkDestroySemaphore DestroySemaphore{};
   PFN_vkDestroyFence DestroyFence{};
   PFN_vkAcquireNextImageKHR AcquireNextImageKHR{};
   PFN_vkResetFences ResetFences{};
   PFN_vkWaitForFences WaitForFences{};
   PFN_vkResetCommandBuffer ResetCommandBuffer{};
   PFN_vkBeginCommandBuffer BeginCommandBuffer{};
   PFN_vkCmdPipelineBarrier CmdPipelineBarrier{};
   PFN_vkCmdBeginRenderPass CmdBeginRenderPass{};
   PFN_vkCmdEndRenderPass CmdEndRenderPass{};
   PFN_vkCmdBindPipeline CmdBindPipeline{};
   PFN_vkCmdSetViewport CmdSetViewport{};
   PFN_vkCmdSetScissor CmdSetScissor{};
   PFN_vkCmdPushConstants CmdPushConstants{};
   PFN_vkCmdDraw CmdDraw{};
   PFN_vkEndCommandBuffer EndCommandBuffer{};
   PFN_vkQueueSubmit QueueSubmit{};
   PFN_vkQueuePresentKHR QueuePresentKHR{};
   PFN_vkDeviceWaitIdle DeviceWaitIdle{};
};

State s;
std::mutex hud_mutex;
std::string hud_text =
   "Present calls 0.0/s\n"
   "CPU present interval 0.00 ms avg\n"
   "Surface 0x0\n"
   "GPU Not active";

void publish_hud(float present_calls_per_second, double interval_ms,
                 uint32_t width, uint32_t height, const char *device_name)
{
   char text[512];
   std::snprintf(text, sizeof(text),
                 "Present calls %.1f/s\n"
                 "CPU present interval %.2f ms avg\n"
                 "Surface %ux%u\n"
                 "GPU %s",
                 present_calls_per_second, interval_ms, width, height,
                 device_name && device_name[0] ? device_name : "Unavailable");
   std::lock_guard<std::mutex> lock(hud_mutex);
   hud_text = text;
}

void reset_hud(const char *device_name)
{
   publish_hud(0.0f, 0.0, 0, 0, device_name);
}

template <typename T>
T loader_proc(const char *name)
{
   return reinterpret_cast<T>(dlsym(s.loader, name));
}

template <typename T>
T instance_proc(const char *name)
{
   return reinterpret_cast<T>(s.GetInstanceProcAddr(s.instance, name));
}

template <typename T>
T device_proc(const char *name)
{
   return reinterpret_cast<T>(s.GetDeviceProcAddr(s.device, name));
}

bool check(VkResult result, const char *operation)
{
   if (result == VK_SUCCESS)
      return true;
   LOGE("%s failed: %d", operation, result);
   return false;
}
} // namespace

bool panvk_buffer_copy_probe(void *libvulkan, const char *label)
{
   constexpr VkDeviceSize size = 4096;
   constexpr VkDeviceSize buffer_size = 8192;
   VkInstance instance{};
   VkDevice device{};
   VkBuffer src{};
   VkBuffer dst{};
   VkDeviceMemory src_memory{};
   VkDeviceMemory dst_memory{};
   VkCommandPool pool{};
   VkDescriptorSetLayout descriptor_layout{};
   VkDescriptorPool descriptor_pool{};
   VkShaderModule compute_shader{};
   VkPipelineLayout compute_pipeline_layout{};
   VkPipeline compute_pipeline{};
   VkImage sampled_image{};
   VkDeviceMemory sampled_image_memory{};
   VkImage output_image{};
   VkDeviceMemory output_image_memory{};
   VkImageView sampled_image_view{};
   VkImageView output_image_view{};
   VkSampler sampler{};
   VkDescriptorSetLayout sample_descriptor_layout{};
   VkDescriptorPool sample_descriptor_pool{};
   VkShaderModule sample_vertex_shader{};
   VkShaderModule sample_fragment_shader{};
   VkPipelineLayout sample_pipeline_layout{};
   VkRenderPass sample_render_pass{};
   VkPipeline sample_pipeline{};
   VkFramebuffer sample_framebuffer{};
   VkImage depth_image{};
   VkDeviceMemory depth_image_memory{};
   VkImageView depth_image_view{};
   VkShaderModule draw_vertex_shader{};
   VkShaderModule draw_fragment_shader{};
   VkPipelineLayout draw_pipeline_layout{};
   VkRenderPass draw_render_pass{};
   VkFramebuffer draw_framebuffer{};
   VkPipeline indexed_pipeline{};
   VkPipeline blend_pipeline{};
   VkPipeline depth_pipeline{};
   void *src_map{};
   void *dst_map{};

   auto GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(libvulkan, "vkGetInstanceProcAddr"));
   auto CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      dlsym(libvulkan, "vkCreateInstance"));
   if (!GetInstanceProcAddr || !CreateInstance) {
      LOGE("buffer-copy[%s]: loader entrypoints unavailable", label);
      return false;
   }

   VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
   app_info.pApplicationName = "PanVK buffer copy probe";
   app_info.apiVersion = VK_API_VERSION_1_0;
   VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   instance_info.pApplicationInfo = &app_info;
   VkResult result = CreateInstance(&instance_info, nullptr, &instance);
   if (result != VK_SUCCESS) {
      LOGE("buffer-copy[%s]: vkCreateInstance failed: %d", label, result);
      return false;
   }

#define INSTANCE_PROC(type, name) \
   auto name = reinterpret_cast<type>(GetInstanceProcAddr(instance, "vk" #name))
   INSTANCE_PROC(PFN_vkDestroyInstance, DestroyInstance);
   INSTANCE_PROC(PFN_vkEnumeratePhysicalDevices, EnumeratePhysicalDevices);
   INSTANCE_PROC(PFN_vkGetPhysicalDeviceQueueFamilyProperties,
                 GetPhysicalDeviceQueueFamilyProperties);
   INSTANCE_PROC(PFN_vkGetPhysicalDeviceMemoryProperties,
                 GetPhysicalDeviceMemoryProperties);
   INSTANCE_PROC(PFN_vkGetPhysicalDeviceImageFormatProperties,
                 GetPhysicalDeviceImageFormatProperties);
   INSTANCE_PROC(PFN_vkCreateDevice, CreateDevice);
   INSTANCE_PROC(PFN_vkGetDeviceProcAddr, GetDeviceProcAddr);
#undef INSTANCE_PROC

   if (!DestroyInstance || !EnumeratePhysicalDevices ||
       !GetPhysicalDeviceQueueFamilyProperties ||
       !GetPhysicalDeviceMemoryProperties ||
       !GetPhysicalDeviceImageFormatProperties || !CreateDevice ||
       !GetDeviceProcAddr) {
      LOGE("buffer-copy[%s]: instance entrypoints unavailable", label);
      if (DestroyInstance)
         DestroyInstance(instance, nullptr);
      return false;
   }

   uint32_t physical_count = 0;
   result = EnumeratePhysicalDevices(instance, &physical_count, nullptr);
   std::vector<VkPhysicalDevice> physical_devices(physical_count);
   if (result == VK_SUCCESS && physical_count)
      result = EnumeratePhysicalDevices(instance, &physical_count,
                                        physical_devices.data());
   if (result != VK_SUCCESS || physical_count == 0) {
      LOGE("buffer-copy[%s]: no physical device: %d", label, result);
      DestroyInstance(instance, nullptr);
      return false;
   }
   VkPhysicalDevice physical = physical_devices[0];

   uint32_t family_count = 0;
   GetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
   std::vector<VkQueueFamilyProperties> families(family_count);
   GetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                          families.data());
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueCount &&
          (families[i].queueFlags &
           (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
             (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
         queue_family = i;
         break;
      }
   }
   if (queue_family == UINT32_MAX) {
      LOGI("buffer-copy[%s]: SKIP unsupported graphics+compute queue", label);
      DestroyInstance(instance, nullptr);
      return false;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
   queue_info.queueFamilyIndex = queue_family;
   queue_info.queueCount = 1;
   queue_info.pQueuePriorities = &priority;
   VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
   device_info.queueCreateInfoCount = 1;
   device_info.pQueueCreateInfos = &queue_info;
   result = CreateDevice(physical, &device_info, nullptr, &device);
   if (result != VK_SUCCESS) {
      LOGE("buffer-copy[%s]: vkCreateDevice failed: %d", label, result);
      DestroyInstance(instance, nullptr);
      return false;
   }

#define DEVICE_PROC(type, name) \
   auto name = reinterpret_cast<type>(GetDeviceProcAddr(device, "vk" #name))
   DEVICE_PROC(PFN_vkDestroyDevice, DestroyDevice);
   DEVICE_PROC(PFN_vkGetDeviceQueue, GetDeviceQueue);
   DEVICE_PROC(PFN_vkCreateBuffer, CreateBuffer);
   DEVICE_PROC(PFN_vkDestroyBuffer, DestroyBuffer);
   DEVICE_PROC(PFN_vkGetBufferMemoryRequirements, GetBufferMemoryRequirements);
   DEVICE_PROC(PFN_vkAllocateMemory, AllocateMemory);
   DEVICE_PROC(PFN_vkFreeMemory, FreeMemory);
   DEVICE_PROC(PFN_vkBindBufferMemory, BindBufferMemory);
   DEVICE_PROC(PFN_vkMapMemory, MapMemory);
   DEVICE_PROC(PFN_vkUnmapMemory, UnmapMemory);
   DEVICE_PROC(PFN_vkFlushMappedMemoryRanges, FlushMappedMemoryRanges);
   DEVICE_PROC(PFN_vkInvalidateMappedMemoryRanges, InvalidateMappedMemoryRanges);
   DEVICE_PROC(PFN_vkCreateCommandPool, CreateCommandPool);
   DEVICE_PROC(PFN_vkDestroyCommandPool, DestroyCommandPool);
   DEVICE_PROC(PFN_vkAllocateCommandBuffers, AllocateCommandBuffers);
   DEVICE_PROC(PFN_vkResetCommandBuffer, ResetCommandBuffer);
   DEVICE_PROC(PFN_vkBeginCommandBuffer, BeginCommandBuffer);
   DEVICE_PROC(PFN_vkCmdCopyBuffer, CmdCopyBuffer);
   DEVICE_PROC(PFN_vkCreateImage, CreateImage);
   DEVICE_PROC(PFN_vkDestroyImage, DestroyImage);
   DEVICE_PROC(PFN_vkGetImageMemoryRequirements, GetImageMemoryRequirements);
   DEVICE_PROC(PFN_vkBindImageMemory, BindImageMemory);
   DEVICE_PROC(PFN_vkCmdCopyBufferToImage, CmdCopyBufferToImage);
   DEVICE_PROC(PFN_vkCmdCopyImageToBuffer, CmdCopyImageToBuffer);
   DEVICE_PROC(PFN_vkCreateImageView, CreateImageView);
   DEVICE_PROC(PFN_vkDestroyImageView, DestroyImageView);
   DEVICE_PROC(PFN_vkCreateSampler, CreateSampler);
   DEVICE_PROC(PFN_vkDestroySampler, DestroySampler);
   DEVICE_PROC(PFN_vkCreateDescriptorSetLayout, CreateDescriptorSetLayout);
   DEVICE_PROC(PFN_vkDestroyDescriptorSetLayout, DestroyDescriptorSetLayout);
   DEVICE_PROC(PFN_vkCreateDescriptorPool, CreateDescriptorPool);
   DEVICE_PROC(PFN_vkDestroyDescriptorPool, DestroyDescriptorPool);
   DEVICE_PROC(PFN_vkAllocateDescriptorSets, AllocateDescriptorSets);
   DEVICE_PROC(PFN_vkUpdateDescriptorSets, UpdateDescriptorSets);
   DEVICE_PROC(PFN_vkCreateShaderModule, CreateShaderModule);
   DEVICE_PROC(PFN_vkDestroyShaderModule, DestroyShaderModule);
   DEVICE_PROC(PFN_vkCreatePipelineLayout, CreatePipelineLayout);
   DEVICE_PROC(PFN_vkDestroyPipelineLayout, DestroyPipelineLayout);
   DEVICE_PROC(PFN_vkCreateComputePipelines, CreateComputePipelines);
   DEVICE_PROC(PFN_vkCreateGraphicsPipelines, CreateGraphicsPipelines);
   DEVICE_PROC(PFN_vkDestroyPipeline, DestroyPipeline);
   DEVICE_PROC(PFN_vkCmdBindPipeline, CmdBindPipeline);
   DEVICE_PROC(PFN_vkCmdBindDescriptorSets, CmdBindDescriptorSets);
   DEVICE_PROC(PFN_vkCmdDispatch, CmdDispatch);
   DEVICE_PROC(PFN_vkCmdFillBuffer, CmdFillBuffer);
   DEVICE_PROC(PFN_vkCreateRenderPass, CreateRenderPass);
   DEVICE_PROC(PFN_vkDestroyRenderPass, DestroyRenderPass);
   DEVICE_PROC(PFN_vkCreateFramebuffer, CreateFramebuffer);
   DEVICE_PROC(PFN_vkDestroyFramebuffer, DestroyFramebuffer);
   DEVICE_PROC(PFN_vkCmdBeginRenderPass, CmdBeginRenderPass);
   DEVICE_PROC(PFN_vkCmdEndRenderPass, CmdEndRenderPass);
   DEVICE_PROC(PFN_vkCmdDraw, CmdDraw);
   DEVICE_PROC(PFN_vkCmdBindIndexBuffer, CmdBindIndexBuffer);
   DEVICE_PROC(PFN_vkCmdDrawIndexed, CmdDrawIndexed);
   DEVICE_PROC(PFN_vkCmdPushConstants, CmdPushConstants);
   DEVICE_PROC(PFN_vkCmdPipelineBarrier, CmdPipelineBarrier);
   DEVICE_PROC(PFN_vkEndCommandBuffer, EndCommandBuffer);
   DEVICE_PROC(PFN_vkCreateFence, CreateFence);
   DEVICE_PROC(PFN_vkDestroyFence, DestroyFence);
   DEVICE_PROC(PFN_vkCreateSemaphore, CreateSemaphore);
   DEVICE_PROC(PFN_vkDestroySemaphore, DestroySemaphore);
   DEVICE_PROC(PFN_vkGetFenceStatus, GetFenceStatus);
   DEVICE_PROC(PFN_vkResetFences, ResetFences);
   DEVICE_PROC(PFN_vkWaitForFences, WaitForFences);
   DEVICE_PROC(PFN_vkQueueSubmit, QueueSubmit);
   DEVICE_PROC(PFN_vkQueueWaitIdle, QueueWaitIdle);
   DEVICE_PROC(PFN_vkDeviceWaitIdle, DeviceWaitIdle);
#undef DEVICE_PROC

   auto cleanup = [&]() {
      if (device && DeviceWaitIdle)
         DeviceWaitIdle(device);
      if (src_map && UnmapMemory)
         UnmapMemory(device, src_memory);
      if (dst_map && UnmapMemory)
         UnmapMemory(device, dst_memory);
      if (draw_framebuffer && DestroyFramebuffer)
         DestroyFramebuffer(device, draw_framebuffer, nullptr);
      if (indexed_pipeline && DestroyPipeline)
         DestroyPipeline(device, indexed_pipeline, nullptr);
      if (blend_pipeline && DestroyPipeline)
         DestroyPipeline(device, blend_pipeline, nullptr);
      if (depth_pipeline && DestroyPipeline)
         DestroyPipeline(device, depth_pipeline, nullptr);
      if (draw_render_pass && DestroyRenderPass)
         DestroyRenderPass(device, draw_render_pass, nullptr);
      if (draw_pipeline_layout && DestroyPipelineLayout)
         DestroyPipelineLayout(device, draw_pipeline_layout, nullptr);
      if (draw_vertex_shader && DestroyShaderModule)
         DestroyShaderModule(device, draw_vertex_shader, nullptr);
      if (draw_fragment_shader && DestroyShaderModule)
         DestroyShaderModule(device, draw_fragment_shader, nullptr);
      if (depth_image_view && DestroyImageView)
         DestroyImageView(device, depth_image_view, nullptr);
      if (depth_image && DestroyImage)
         DestroyImage(device, depth_image, nullptr);
      if (depth_image_memory && FreeMemory)
         FreeMemory(device, depth_image_memory, nullptr);
      if (sample_framebuffer && DestroyFramebuffer)
         DestroyFramebuffer(device, sample_framebuffer, nullptr);
      if (sample_pipeline && DestroyPipeline)
         DestroyPipeline(device, sample_pipeline, nullptr);
      if (sample_render_pass && DestroyRenderPass)
         DestroyRenderPass(device, sample_render_pass, nullptr);
      if (sample_pipeline_layout && DestroyPipelineLayout)
         DestroyPipelineLayout(device, sample_pipeline_layout, nullptr);
      if (sample_vertex_shader && DestroyShaderModule)
         DestroyShaderModule(device, sample_vertex_shader, nullptr);
      if (sample_fragment_shader && DestroyShaderModule)
         DestroyShaderModule(device, sample_fragment_shader, nullptr);
      if (sample_descriptor_pool && DestroyDescriptorPool)
         DestroyDescriptorPool(device, sample_descriptor_pool, nullptr);
      if (sample_descriptor_layout && DestroyDescriptorSetLayout)
         DestroyDescriptorSetLayout(device, sample_descriptor_layout, nullptr);
      if (sampler && DestroySampler)
         DestroySampler(device, sampler, nullptr);
      if (sampled_image_view && DestroyImageView)
         DestroyImageView(device, sampled_image_view, nullptr);
      if (output_image_view && DestroyImageView)
         DestroyImageView(device, output_image_view, nullptr);
      if (sampled_image && DestroyImage)
         DestroyImage(device, sampled_image, nullptr);
      if (output_image && DestroyImage)
         DestroyImage(device, output_image, nullptr);
      if (sampled_image_memory && FreeMemory)
         FreeMemory(device, sampled_image_memory, nullptr);
      if (output_image_memory && FreeMemory)
         FreeMemory(device, output_image_memory, nullptr);
      if (compute_pipeline && DestroyPipeline)
         DestroyPipeline(device, compute_pipeline, nullptr);
      if (compute_pipeline_layout && DestroyPipelineLayout)
         DestroyPipelineLayout(device, compute_pipeline_layout, nullptr);
      if (compute_shader && DestroyShaderModule)
         DestroyShaderModule(device, compute_shader, nullptr);
      if (descriptor_pool && DestroyDescriptorPool)
         DestroyDescriptorPool(device, descriptor_pool, nullptr);
      if (descriptor_layout && DestroyDescriptorSetLayout)
         DestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      if (pool && DestroyCommandPool)
         DestroyCommandPool(device, pool, nullptr);
      if (src && DestroyBuffer)
         DestroyBuffer(device, src, nullptr);
      if (dst && DestroyBuffer)
         DestroyBuffer(device, dst, nullptr);
      if (src_memory && FreeMemory)
         FreeMemory(device, src_memory, nullptr);
      if (dst_memory && FreeMemory)
         FreeMemory(device, dst_memory, nullptr);
      if (device && DestroyDevice)
         DestroyDevice(device, nullptr);
      DestroyInstance(instance, nullptr);
   };
   auto failed = [&](const char *operation, VkResult failure) {
      LOGE("buffer-copy[%s]: %s failed: %d", label, operation, failure);
      cleanup();
      return false;
   };

   if (!DestroyDevice || !GetDeviceQueue || !CreateBuffer || !DestroyBuffer ||
       !GetBufferMemoryRequirements || !AllocateMemory || !FreeMemory ||
       !BindBufferMemory || !MapMemory || !UnmapMemory ||
       !FlushMappedMemoryRanges || !InvalidateMappedMemoryRanges ||
       !CreateCommandPool || !DestroyCommandPool || !AllocateCommandBuffers ||
       !ResetCommandBuffer || !BeginCommandBuffer || !CmdCopyBuffer ||
       !CreateImage || !DestroyImage ||
       !GetImageMemoryRequirements || !BindImageMemory ||
       !CmdCopyBufferToImage || !CmdCopyImageToBuffer || !CreateImageView ||
       !DestroyImageView || !CreateSampler || !DestroySampler ||
       !CreateDescriptorSetLayout ||
       !DestroyDescriptorSetLayout || !CreateDescriptorPool ||
       !DestroyDescriptorPool || !AllocateDescriptorSets ||
       !UpdateDescriptorSets || !CreateShaderModule || !DestroyShaderModule ||
       !CreatePipelineLayout || !DestroyPipelineLayout ||
       !CreateComputePipelines || !CreateGraphicsPipelines ||
       !DestroyPipeline || !CmdBindPipeline ||
       !CmdBindDescriptorSets || !CmdDispatch || !CmdFillBuffer ||
       !CmdPipelineBarrier ||
       !CreateRenderPass || !DestroyRenderPass || !CreateFramebuffer ||
       !DestroyFramebuffer || !CmdBeginRenderPass || !CmdEndRenderPass ||
       !CmdDraw || !CmdBindIndexBuffer || !CmdDrawIndexed ||
       !CmdPushConstants || !EndCommandBuffer ||
       !CreateFence || !DestroyFence || !CreateSemaphore ||
       !DestroySemaphore || !GetFenceStatus || !ResetFences ||
       !WaitForFences || !QueueSubmit || !QueueWaitIdle || !DeviceWaitIdle) {
      LOGE("buffer-copy[%s]: device entrypoints unavailable", label);
      cleanup();
      return false;
   }

   VkQueue queue{};
   GetDeviceQueue(device, queue_family, 0, &queue);
   VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
   buffer_info.size = buffer_size;
   buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
   buffer_info.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
   result = CreateBuffer(device, &buffer_info, nullptr, &src);
   if (result != VK_SUCCESS)
      return failed("vkCreateBuffer(src)", result);
   buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
   result = CreateBuffer(device, &buffer_info, nullptr, &dst);
   if (result != VK_SUCCESS)
      return failed("vkCreateBuffer(dst)", result);

   VkMemoryRequirements src_requirements{};
   VkMemoryRequirements dst_requirements{};
   GetBufferMemoryRequirements(device, src, &src_requirements);
   GetBufferMemoryRequirements(device, dst, &dst_requirements);
   VkPhysicalDeviceMemoryProperties memory_properties{};
   GetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   const uint32_t supported_types =
      src_requirements.memoryTypeBits & dst_requirements.memoryTypeBits;
   uint32_t memory_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      const VkMemoryPropertyFlags flags =
         memory_properties.memoryTypes[i].propertyFlags;
      if ((supported_types & (1u << i)) &&
          (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          !(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         memory_type = i;
         break;
      }
   }
   if (memory_type == UINT32_MAX) {
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
         if ((supported_types & (1u << i)) &&
             (memory_properties.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memory_type = i;
            break;
         }
      }
   }
   if (memory_type == UINT32_MAX) {
      LOGE("buffer-copy[%s]: no common host-visible memory type", label);
      cleanup();
      return false;
   }
   const VkMemoryPropertyFlags memory_flags =
      memory_properties.memoryTypes[memory_type].propertyFlags;

   VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
   alloc_info.memoryTypeIndex = memory_type;
   alloc_info.allocationSize = src_requirements.size;
   result = AllocateMemory(device, &alloc_info, nullptr, &src_memory);
   if (result != VK_SUCCESS)
      return failed("vkAllocateMemory(src)", result);
   alloc_info.allocationSize = dst_requirements.size;
   result = AllocateMemory(device, &alloc_info, nullptr, &dst_memory);
   if (result != VK_SUCCESS)
      return failed("vkAllocateMemory(dst)", result);
   result = BindBufferMemory(device, src, src_memory, 0);
   if (result != VK_SUCCESS)
      return failed("vkBindBufferMemory(src)", result);
   result = BindBufferMemory(device, dst, dst_memory, 0);
   if (result != VK_SUCCESS)
      return failed("vkBindBufferMemory(dst)", result);
   result = MapMemory(device, src_memory, 0, VK_WHOLE_SIZE, 0, &src_map);
   if (result != VK_SUCCESS)
      return failed("vkMapMemory(src)", result);
   result = MapMemory(device, dst_memory, 0, VK_WHOLE_SIZE, 0, &dst_map);
   if (result != VK_SUCCESS)
      return failed("vkMapMemory(dst)", result);

   std::array<uint32_t, size / sizeof(uint32_t)> expected{};
   for (uint32_t i = 0; i < expected.size(); ++i)
      expected[i] = (0x9e3779b9u * (i + 1)) ^ 0xa5a55a5au;
   std::memcpy(src_map, expected.data(), size);
   std::memset(dst_map, 0x5a, size);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange ranges[2] = {
         {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, src_memory, 0,
          VK_WHOLE_SIZE},
         {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, dst_memory, 0,
          VK_WHOLE_SIZE},
      };
      result = FlushMappedMemoryRanges(device, 2, ranges);
      if (result != VK_SUCCESS)
         return failed("vkFlushMappedMemoryRanges", result);
   }

   VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   pool_info.queueFamilyIndex = queue_family;
   result = CreateCommandPool(device, &pool_info, nullptr, &pool);
   if (result != VK_SUCCESS)
      return failed("vkCreateCommandPool", result);
   VkCommandBuffer command{};
   VkCommandBufferAllocateInfo command_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
   command_info.commandPool = pool;
   command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   command_info.commandBufferCount = 1;
   result = AllocateCommandBuffers(device, &command_info, &command);
   if (result != VK_SUCCESS)
      return failed("vkAllocateCommandBuffers", result);
   VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   result = BeginCommandBuffer(command, &begin_info);
   if (result != VK_SUCCESS)
      return failed("vkBeginCommandBuffer", result);
   VkBufferCopy copy{0, 0, size};
   CmdCopyBuffer(command, src, dst, 1, &copy);
   result = EndCommandBuffer(command);
   if (result != VK_SUCCESS)
      return failed("vkEndCommandBuffer", result);
   VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
   submit.commandBufferCount = 1;
   submit.pCommandBuffers = &command;
   result = QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
   if (result != VK_SUCCESS)
      return failed("vkQueueSubmit", result);
   result = QueueWaitIdle(queue);
   if (result != VK_SUCCESS)
      return failed("vkQueueWaitIdle", result);

   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = InvalidateMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("vkInvalidateMappedMemoryRanges", result);
   }

   uint64_t expected_hash = 1469598103934665603ull;
   uint64_t actual_hash = 1469598103934665603ull;
   const auto *expected_bytes = reinterpret_cast<const uint8_t *>(expected.data());
   const auto *actual_bytes = reinterpret_cast<const uint8_t *>(dst_map);
   for (VkDeviceSize i = 0; i < size; ++i) {
      expected_hash = (expected_hash ^ expected_bytes[i]) * 1099511628211ull;
      actual_hash = (actual_hash ^ actual_bytes[i]) * 1099511628211ull;
   }
   if (std::memcmp(expected.data(), dst_map, size) != 0) {
      VkDeviceSize first = 0;
      while (first < size && expected_bytes[first] == actual_bytes[first])
         ++first;
      LOGE("buffer-copy[%s]: FAIL byte=%llu expected=%02x actual=%02x "
           "expected_hash=%016llx actual_hash=%016llx",
           label, static_cast<unsigned long long>(first), expected_bytes[first],
           actual_bytes[first], static_cast<unsigned long long>(expected_hash),
           static_cast<unsigned long long>(actual_hash));
      cleanup();
      return false;
   }

   LOGI("buffer-copy[%s]: PASS bytes=%llu hash=%016llx memory_type=%u flags=0x%x",
        label, static_cast<unsigned long long>(size),
        static_cast<unsigned long long>(actual_hash), memory_type, memory_flags);

   std::array<uint32_t, size / sizeof(uint32_t)> compute_expected{};
   for (uint32_t i = 0; i < compute_expected.size(); ++i)
      compute_expected[i] = (0x045d9f3bu * (i + 3u)) ^ 0xc001d00du;
   std::memset(dst_map, 0, size);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = FlushMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("compute vkFlushMappedMemoryRanges", result);
   }

   VkDescriptorSetLayoutBinding descriptor_binding{};
   descriptor_binding.binding = 0;
   descriptor_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   descriptor_binding.descriptorCount = 1;
   descriptor_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
   VkDescriptorSetLayoutCreateInfo descriptor_layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
   descriptor_layout_info.bindingCount = 1;
   descriptor_layout_info.pBindings = &descriptor_binding;
   result = CreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr,
                                      &descriptor_layout);
   if (result != VK_SUCCESS)
      return failed("compute vkCreateDescriptorSetLayout", result);

   VkDescriptorPoolSize descriptor_pool_size{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo descriptor_pool_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
   descriptor_pool_info.maxSets = 1;
   descriptor_pool_info.poolSizeCount = 1;
   descriptor_pool_info.pPoolSizes = &descriptor_pool_size;
   result = CreateDescriptorPool(device, &descriptor_pool_info, nullptr,
                                 &descriptor_pool);
   if (result != VK_SUCCESS)
      return failed("compute vkCreateDescriptorPool", result);

   VkDescriptorSet descriptor_set{};
   VkDescriptorSetAllocateInfo descriptor_allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
   descriptor_allocate_info.descriptorPool = descriptor_pool;
   descriptor_allocate_info.descriptorSetCount = 1;
   descriptor_allocate_info.pSetLayouts = &descriptor_layout;
   result = AllocateDescriptorSets(device, &descriptor_allocate_info,
                                   &descriptor_set);
   if (result != VK_SUCCESS)
      return failed("compute vkAllocateDescriptorSets", result);

   VkDescriptorBufferInfo descriptor_buffer{dst, 0, size};
   VkWriteDescriptorSet descriptor_write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
   descriptor_write.dstSet = descriptor_set;
   descriptor_write.dstBinding = 0;
   descriptor_write.descriptorCount = 1;
   descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   descriptor_write.pBufferInfo = &descriptor_buffer;
   UpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);

   VkShaderModuleCreateInfo shader_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
   shader_info.codeSize = sizeof(panvk_compute_spv);
   shader_info.pCode = panvk_compute_spv;
   result = CreateShaderModule(device, &shader_info, nullptr, &compute_shader);
   if (result != VK_SUCCESS)
      return failed("compute vkCreateShaderModule", result);

   VkPipelineLayoutCreateInfo pipeline_layout_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   pipeline_layout_info.setLayoutCount = 1;
   pipeline_layout_info.pSetLayouts = &descriptor_layout;
   result = CreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                 &compute_pipeline_layout);
   if (result != VK_SUCCESS)
      return failed("compute vkCreatePipelineLayout", result);

   VkPipelineShaderStageCreateInfo shader_stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
   shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
   shader_stage.module = compute_shader;
   shader_stage.pName = "main";
   VkComputePipelineCreateInfo compute_pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
   compute_pipeline_info.stage = shader_stage;
   compute_pipeline_info.layout = compute_pipeline_layout;
   result = CreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                   &compute_pipeline_info, nullptr,
                                   &compute_pipeline);
   if (result != VK_SUCCESS)
      return failed("compute vkCreateComputePipelines", result);

   VkCommandBuffer compute_command{};
   result = AllocateCommandBuffers(device, &command_info, &compute_command);
   if (result != VK_SUCCESS)
      return failed("compute vkAllocateCommandBuffers", result);
   result = BeginCommandBuffer(compute_command, &begin_info);
   if (result != VK_SUCCESS)
      return failed("compute vkBeginCommandBuffer", result);
   CmdBindPipeline(compute_command, VK_PIPELINE_BIND_POINT_COMPUTE,
                   compute_pipeline);
   CmdBindDescriptorSets(compute_command, VK_PIPELINE_BIND_POINT_COMPUTE,
                         compute_pipeline_layout, 0, 1, &descriptor_set, 0,
                         nullptr);
   CmdDispatch(compute_command, compute_expected.size() / 64, 1, 1);
   VkBufferMemoryBarrier host_barrier{
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
   host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
   host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   host_barrier.buffer = dst;
   host_barrier.size = size;
   CmdPipelineBarrier(compute_command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                      &host_barrier, 0, nullptr);
   result = EndCommandBuffer(compute_command);
   if (result != VK_SUCCESS)
      return failed("compute vkEndCommandBuffer", result);
   submit.pCommandBuffers = &compute_command;
   result = QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
   if (result != VK_SUCCESS)
      return failed("compute vkQueueSubmit", result);
   result = QueueWaitIdle(queue);
   if (result != VK_SUCCESS)
      return failed("compute vkQueueWaitIdle", result);

   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = InvalidateMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("compute vkInvalidateMappedMemoryRanges", result);
   }

   uint64_t compute_expected_hash = 1469598103934665603ull;
   uint64_t compute_actual_hash = 1469598103934665603ull;
   const auto *compute_expected_bytes =
      reinterpret_cast<const uint8_t *>(compute_expected.data());
   for (VkDeviceSize i = 0; i < size; ++i) {
      compute_expected_hash =
         (compute_expected_hash ^ compute_expected_bytes[i]) *
         1099511628211ull;
      compute_actual_hash = (compute_actual_hash ^ actual_bytes[i]) *
                            1099511628211ull;
   }
   if (std::memcmp(compute_expected.data(), dst_map, size) != 0) {
      VkDeviceSize first = 0;
      while (first < size &&
             compute_expected_bytes[first] == actual_bytes[first])
         ++first;
      LOGE("compute[%s]: FAIL byte=%llu expected=%02x actual=%02x "
           "expected_hash=%016llx actual_hash=%016llx",
           label, static_cast<unsigned long long>(first),
           compute_expected_bytes[first], actual_bytes[first],
           static_cast<unsigned long long>(compute_expected_hash),
           static_cast<unsigned long long>(compute_actual_hash));
      cleanup();
      return false;
   }

   LOGI("compute[%s]: PASS invocations=%zu hash=%016llx memory_type=%u "
        "flags=0x%x",
        label, compute_expected.size(),
        static_cast<unsigned long long>(compute_actual_hash), memory_type,
        memory_flags);

   enum class LifecycleCycleResult {
      pass,
      fail_safe,
   };
   struct LifecycleResourceSnapshot {
      long fds = -1;
      long maps = -1;
      long rss_kb = -1;
      long size_kb = -1;
      uint64_t mapped_bytes = UINT64_MAX;
   };
   auto lifecycle_resource_snapshot = []() {
      LifecycleResourceSnapshot snapshot;
      if (DIR *directory = opendir("/proc/self/fd")) {
         snapshot.fds = 0;
         while (dirent *entry = readdir(directory)) {
            if (std::strcmp(entry->d_name, ".") &&
                std::strcmp(entry->d_name, ".."))
               ++snapshot.fds;
         }
         closedir(directory);
      }
      if (FILE *maps = std::fopen("/proc/self/maps", "r")) {
         snapshot.maps = 0;
         snapshot.mapped_bytes = 0;
         bool valid = true;
         char *line = nullptr;
         size_t capacity = 0;
         while (getline(&line, &capacity, maps) >= 0) {
            unsigned long long start = 0;
            unsigned long long end = 0;
            ++snapshot.maps;
            if (std::sscanf(line, "%llx-%llx", &start, &end) != 2 ||
                end < start ||
                UINT64_MAX - snapshot.mapped_bytes < end - start) {
               valid = false;
               continue;
            }
            snapshot.mapped_bytes += end - start;
         }
         std::free(line);
         if (std::ferror(maps) || !valid) {
            snapshot.maps = -1;
            snapshot.mapped_bytes = UINT64_MAX;
         }
         std::fclose(maps);
      }
      if (FILE *status = std::fopen("/proc/self/status", "r")) {
         char line[256];
         while (std::fgets(line, sizeof(line), status)) {
            std::sscanf(line, "VmRSS: %ld kB", &snapshot.rss_kb);
            std::sscanf(line, "VmSize: %ld kB", &snapshot.size_kb);
         }
         std::fclose(status);
      }
      return snapshot;
   };
   auto lifecycle_snapshot_complete =
      [](const LifecycleResourceSnapshot &snapshot) {
         return snapshot.fds >= 0 && snapshot.maps >= 0 &&
                snapshot.rss_kb >= 0 && snapshot.size_kb >= 0 &&
                snapshot.mapped_bytes != UINT64_MAX;
      };
   auto lifecycle_ownership_equal =
      [](const LifecycleResourceSnapshot &left,
         const LifecycleResourceSnapshot &right) {
         return left.fds == right.fds && left.maps == right.maps &&
                left.mapped_bytes == right.mapped_bytes &&
                left.size_kb == right.size_kb;
      };

   constexpr VkDeviceSize lifecycle_buffer_size = 4096;
   constexpr uint32_t lifecycle_dispatch_groups = 15;
   constexpr uint32_t lifecycle_shader_words =
      lifecycle_dispatch_groups * 64;
   constexpr uint32_t lifecycle_sentinel_zero_index = lifecycle_shader_words;
   constexpr uint32_t lifecycle_sentinel_one_index =
      lifecycle_shader_words + 1;
   constexpr uint32_t lifecycle_guard_front_index =
      lifecycle_shader_words + 2;
   constexpr uint32_t lifecycle_guard_back_index =
      lifecycle_buffer_size / sizeof(uint32_t) - 1;
   constexpr uint32_t lifecycle_guard_front = 0x13579bdfu;
   constexpr uint32_t lifecycle_guard_back = 0xfdb97531u;
   constexpr uint64_t lifecycle_fence_timeout_ns =
      5ull * 1000 * 1000 * 1000;
   static_assert(lifecycle_guard_front_index < lifecycle_guard_back_index,
                 "lifecycle sentinels must be outside shader writes");

   auto lifecycle_cycle = [&](const char *phase,
                              uint32_t cycle) -> LifecycleCycleResult {
      VkBuffer cycle_buffer{};
      VkDeviceMemory cycle_memory{};
      void *cycle_map{};
      VkDescriptorPool cycle_descriptor_pool{};
      VkCommandPool cycle_command_pool{};
      VkSemaphore cycle_semaphore{};
      VkFence cycle_fence{};
      bool submitted = false;
      bool completed = false;

      auto release_cycle = [&]() {
         if (cycle_map)
            UnmapMemory(device, cycle_memory);
         if (cycle_fence)
            DestroyFence(device, cycle_fence, nullptr);
         if (cycle_semaphore)
            DestroySemaphore(device, cycle_semaphore, nullptr);
         if (cycle_command_pool)
            DestroyCommandPool(device, cycle_command_pool, nullptr);
         if (cycle_descriptor_pool)
            DestroyDescriptorPool(device, cycle_descriptor_pool, nullptr);
         if (cycle_buffer)
            DestroyBuffer(device, cycle_buffer, nullptr);
         if (cycle_memory)
            FreeMemory(device, cycle_memory, nullptr);
      };
      auto functional_fail = [&](const char *operation,
                                 VkResult failure) {
         LOGE("lifecycle-churn[%s]: result=FUNCTIONAL_FAIL phase=%s "
              "cycle=%u operation=%s vk_result=%d submitted=%u "
              "completed=%u",
              label, phase, cycle, operation, failure, submitted, completed);
         release_cycle();
         return LifecycleCycleResult::fail_safe;
      };
      auto completion_inconclusive = [&](const char *operation,
                                         VkResult failure)
         -> LifecycleCycleResult {
         LOGE("lifecycle-churn[%s]: result=INCONCLUSIVE "
              "failure=FUNCTIONAL phase=%s cycle=%u operation=%s "
              "vk_result=%d submitted=1 completed=0 "
              "reason=completion-uncertain action=terminate-process "
              "exit_code=124 resources=retained",
              label, phase, cycle, operation, failure);
         _exit(124);
      };

      VkBufferCreateInfo cycle_buffer_info{
         VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      cycle_buffer_info.size = lifecycle_buffer_size;
      cycle_buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      cycle_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VkResult cycle_result =
         CreateBuffer(device, &cycle_buffer_info, nullptr, &cycle_buffer);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkCreateBuffer", cycle_result);

      VkMemoryRequirements cycle_requirements{};
      GetBufferMemoryRequirements(device, cycle_buffer, &cycle_requirements);
      uint32_t cycle_memory_type = UINT32_MAX;
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
         if ((cycle_requirements.memoryTypeBits & (1u << i)) &&
             (memory_properties.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            cycle_memory_type = i;
            break;
         }
      }
      if (cycle_memory_type == UINT32_MAX)
         return functional_fail("host-visible-memory-type",
                                VK_ERROR_FEATURE_NOT_PRESENT);
      const VkMemoryPropertyFlags cycle_memory_flags =
         memory_properties.memoryTypes[cycle_memory_type].propertyFlags;
      VkMemoryAllocateInfo cycle_alloc_info{
         VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      cycle_alloc_info.allocationSize = cycle_requirements.size;
      cycle_alloc_info.memoryTypeIndex = cycle_memory_type;
      cycle_result =
         AllocateMemory(device, &cycle_alloc_info, nullptr, &cycle_memory);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkAllocateMemory", cycle_result);
      cycle_result = BindBufferMemory(device, cycle_buffer, cycle_memory, 0);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkBindBufferMemory", cycle_result);
      cycle_result = MapMemory(device, cycle_memory, 0, VK_WHOLE_SIZE, 0,
                               &cycle_map);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkMapMemory", cycle_result);

      const uint32_t phase_tag =
         std::strcmp(phase, "measured") == 0 ? 0x00010000u : 0u;
      const uint32_t padding = 0xa11c0000u ^ phase_tag ^ cycle;
      const uint32_t expected_sentinel_zero =
         0xc1c00000u ^ phase_tag ^ cycle;
      const uint32_t expected_sentinel_one =
         0x5e1e0000u ^ phase_tag ^ cycle;
      auto *cycle_words = static_cast<uint32_t *>(cycle_map);
      std::fill_n(cycle_words,
                  lifecycle_buffer_size / sizeof(uint32_t), padding);
      cycle_words[lifecycle_sentinel_zero_index] = 0xbaadf00du;
      cycle_words[lifecycle_sentinel_one_index] = 0xdead10ccu;
      cycle_words[lifecycle_guard_front_index] = lifecycle_guard_front;
      cycle_words[lifecycle_guard_back_index] = lifecycle_guard_back;
      if (!(cycle_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = cycle_memory;
         range.size = VK_WHOLE_SIZE;
         cycle_result = FlushMappedMemoryRanges(device, 1, &range);
         if (cycle_result != VK_SUCCESS)
            return functional_fail("vkFlushMappedMemoryRanges", cycle_result);
      }

      VkDescriptorPoolSize cycle_pool_size{
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
      VkDescriptorPoolCreateInfo cycle_descriptor_pool_info{
         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      cycle_descriptor_pool_info.maxSets = 1;
      cycle_descriptor_pool_info.poolSizeCount = 1;
      cycle_descriptor_pool_info.pPoolSizes = &cycle_pool_size;
      cycle_result = CreateDescriptorPool(device, &cycle_descriptor_pool_info,
                                          nullptr, &cycle_descriptor_pool);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkCreateDescriptorPool", cycle_result);
      VkDescriptorSet cycle_descriptor_set{};
      VkDescriptorSetAllocateInfo cycle_descriptor_allocate_info{
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      cycle_descriptor_allocate_info.descriptorPool = cycle_descriptor_pool;
      cycle_descriptor_allocate_info.descriptorSetCount = 1;
      cycle_descriptor_allocate_info.pSetLayouts = &descriptor_layout;
      cycle_result = AllocateDescriptorSets(
         device, &cycle_descriptor_allocate_info, &cycle_descriptor_set);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkAllocateDescriptorSets", cycle_result);
      VkDescriptorBufferInfo cycle_descriptor_buffer{
         cycle_buffer, 0, lifecycle_buffer_size};
      VkWriteDescriptorSet cycle_descriptor_write{
         VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      cycle_descriptor_write.dstSet = cycle_descriptor_set;
      cycle_descriptor_write.dstBinding = 0;
      cycle_descriptor_write.descriptorCount = 1;
      cycle_descriptor_write.descriptorType =
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      cycle_descriptor_write.pBufferInfo = &cycle_descriptor_buffer;
      UpdateDescriptorSets(device, 1, &cycle_descriptor_write, 0, nullptr);

      VkCommandPoolCreateInfo cycle_command_pool_info{
         VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      cycle_command_pool_info.queueFamilyIndex = queue_family;
      cycle_result = CreateCommandPool(device, &cycle_command_pool_info,
                                       nullptr, &cycle_command_pool);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkCreateCommandPool", cycle_result);
      VkCommandBuffer cycle_commands[2]{};
      VkCommandBufferAllocateInfo cycle_command_allocate_info{
         VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      cycle_command_allocate_info.commandPool = cycle_command_pool;
      cycle_command_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      cycle_command_allocate_info.commandBufferCount = 2;
      cycle_result = AllocateCommandBuffers(device,
                                            &cycle_command_allocate_info,
                                            cycle_commands);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkAllocateCommandBuffers", cycle_result);
      for (uint32_t i = 0; i < 2; ++i) {
         VkCommandBufferBeginInfo cycle_begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
         cycle_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
         cycle_result = BeginCommandBuffer(cycle_commands[i],
                                           &cycle_begin_info);
         if (cycle_result != VK_SUCCESS)
            return functional_fail("vkBeginCommandBuffer", cycle_result);
         CmdBindPipeline(cycle_commands[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                         compute_pipeline);
         CmdBindDescriptorSets(cycle_commands[i],
                               VK_PIPELINE_BIND_POINT_COMPUTE,
                               compute_pipeline_layout, 0, 1,
                               &cycle_descriptor_set, 0, nullptr);
         CmdDispatch(cycle_commands[i], lifecycle_dispatch_groups, 1, 1);
         const VkDeviceSize sentinel_offset =
            static_cast<VkDeviceSize>(i == 0
                                         ? lifecycle_sentinel_zero_index
                                         : lifecycle_sentinel_one_index) *
            sizeof(uint32_t);
         CmdFillBuffer(cycle_commands[i], cycle_buffer, sentinel_offset,
                       sizeof(uint32_t),
                       i == 0 ? expected_sentinel_zero
                              : expected_sentinel_one);
         if (i == 1) {
            VkBufferMemoryBarrier host_barrier{
               VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                         VK_ACCESS_TRANSFER_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.buffer = cycle_buffer;
            host_barrier.size = lifecycle_buffer_size;
            CmdPipelineBarrier(cycle_commands[i],
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                               &host_barrier, 0, nullptr);
         }
         cycle_result = EndCommandBuffer(cycle_commands[i]);
         if (cycle_result != VK_SUCCESS)
            return functional_fail("vkEndCommandBuffer", cycle_result);
      }

      VkSemaphoreCreateInfo semaphore_info{
         VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      cycle_result = CreateSemaphore(device, &semaphore_info, nullptr,
                                     &cycle_semaphore);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkCreateSemaphore", cycle_result);
      VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      cycle_result = CreateFence(device, &fence_info, nullptr, &cycle_fence);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkCreateFence", cycle_result);
      cycle_result = WaitForFences(device, 1, &cycle_fence, VK_TRUE, 0);
      if (cycle_result != VK_TIMEOUT)
         return functional_fail("fresh-fence-timeout-zero", cycle_result);

      VkSubmitInfo first_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      first_submit.commandBufferCount = 1;
      first_submit.pCommandBuffers = &cycle_commands[0];
      first_submit.signalSemaphoreCount = 1;
      first_submit.pSignalSemaphores = &cycle_semaphore;
      cycle_result = QueueSubmit(queue, 1, &first_submit, VK_NULL_HANDLE);
      if (cycle_result != VK_SUCCESS)
         return functional_fail("vkQueueSubmit(first)", cycle_result);
      submitted = true;

      VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkSubmitInfo second_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      second_submit.waitSemaphoreCount = 1;
      second_submit.pWaitSemaphores = &cycle_semaphore;
      second_submit.pWaitDstStageMask = &wait_stage;
      second_submit.commandBufferCount = 1;
      second_submit.pCommandBuffers = &cycle_commands[1];
      cycle_result = QueueSubmit(queue, 1, &second_submit, cycle_fence);
      if (cycle_result != VK_SUCCESS)
         return completion_inconclusive("vkQueueSubmit(second)",
                                        cycle_result);
      cycle_result = WaitForFences(device, 1, &cycle_fence, VK_TRUE,
                                   lifecycle_fence_timeout_ns);
      if (cycle_result != VK_SUCCESS)
         return completion_inconclusive("vkWaitForFences", cycle_result);
      completed = true;

      if (!(cycle_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = cycle_memory;
         range.size = VK_WHOLE_SIZE;
         cycle_result = InvalidateMappedMemoryRanges(device, 1, &range);
         if (cycle_result != VK_SUCCESS)
            return functional_fail("vkInvalidateMappedMemoryRanges",
                                   cycle_result);
      }

      for (uint32_t i = 0; i < lifecycle_shader_words; ++i) {
         const uint32_t expected_value =
            (0x045d9f3bu * (i + 3u)) ^ 0xc001d00du;
         if (cycle_words[i] != expected_value) {
            LOGE("lifecycle-churn[%s]: result=FUNCTIONAL_FAIL phase=%s "
                 "cycle=%u reason=data-mismatch word=%u expected=%08x "
                 "actual=%08x submitted=1 completed=1",
                 label, phase, cycle, i, expected_value, cycle_words[i]);
            release_cycle();
            return LifecycleCycleResult::fail_safe;
         }
      }
      if (cycle_words[lifecycle_sentinel_zero_index] !=
             expected_sentinel_zero ||
          cycle_words[lifecycle_sentinel_one_index] !=
             expected_sentinel_one ||
          cycle_words[lifecycle_guard_front_index] != lifecycle_guard_front ||
          cycle_words[lifecycle_guard_back_index] != lifecycle_guard_back) {
         LOGE("lifecycle-churn[%s]: result=FUNCTIONAL_FAIL phase=%s "
              "cycle=%u reason=sentinel-or-guard-mismatch "
              "sentinel0=%08x/%08x sentinel1=%08x/%08x "
              "front=%08x/%08x back=%08x/%08x submitted=1 completed=1",
              label, phase, cycle,
              cycle_words[lifecycle_sentinel_zero_index],
              expected_sentinel_zero,
              cycle_words[lifecycle_sentinel_one_index],
              expected_sentinel_one,
              cycle_words[lifecycle_guard_front_index],
              lifecycle_guard_front,
              cycle_words[lifecycle_guard_back_index], lifecycle_guard_back);
         release_cycle();
         return LifecycleCycleResult::fail_safe;
      }
      for (uint32_t i = lifecycle_guard_front_index + 1;
           i < lifecycle_guard_back_index; ++i) {
         if (cycle_words[i] != padding) {
            LOGE("lifecycle-churn[%s]: result=FUNCTIONAL_FAIL phase=%s "
                 "cycle=%u reason=padding-mismatch word=%u expected=%08x "
                 "actual=%08x submitted=1 completed=1",
                 label, phase, cycle, i, padding, cycle_words[i]);
            release_cycle();
            return LifecycleCycleResult::fail_safe;
         }
      }

      release_cycle();
      return LifecycleCycleResult::pass;
   };

   const bool lifecycle_dut = std::strcmp(label, "panvk") == 0;
   const char *lifecycle_record =
      lifecycle_dut ? "RESOURCE_GATE" : "RESOURCE_CONTROL";
   const char *lifecycle_role = lifecycle_dut ? "DUT" : "CONTROL";
   const char *lifecycle_gate = lifecycle_dut ? "EXACT" : "DIAGNOSTIC";
   auto lifecycle_log_snapshot = [&](const char *result_name,
                                     const char *phase, uint32_t cycle,
                                     const LifecycleResourceSnapshot &baseline,
                                     const LifecycleResourceSnapshot &current) {
      __android_log_print(
         lifecycle_dut && std::strcmp(result_name, "PASS_STABLE")
            ? ANDROID_LOG_ERROR
            : ANDROID_LOG_INFO,
         "PanvkPresent",
         "lifecycle-churn[%s]: %s result=%s role=%s gate=%s phase=%s "
         "cycle=%u fd=%ld->%ld maps=%ld->%ld mapped_bytes=%llu->%llu "
         "size_kb=%ld->%ld rss_kb=%ld->%ld rss_gate=diagnostic",
         label, lifecycle_record, result_name, lifecycle_role, lifecycle_gate,
         phase, cycle, baseline.fds, current.fds, baseline.maps, current.maps,
         static_cast<unsigned long long>(baseline.mapped_bytes),
         static_cast<unsigned long long>(current.mapped_bytes),
         baseline.size_kb, current.size_kb, baseline.rss_kb, current.rss_kb);
   };

   const LifecycleResourceSnapshot lifecycle_proc_warm_one =
      lifecycle_resource_snapshot();
   const LifecycleResourceSnapshot lifecycle_proc_warm_two =
      lifecycle_resource_snapshot();
   if (!lifecycle_snapshot_complete(lifecycle_proc_warm_one) ||
       !lifecycle_snapshot_complete(lifecycle_proc_warm_two)) {
      lifecycle_log_snapshot("INCONCLUSIVE", "proc-warm", 0,
                             lifecycle_proc_warm_one,
                             lifecycle_proc_warm_two);
      if (lifecycle_dut) {
         cleanup();
         return false;
      }
   } else {
      constexpr uint32_t lifecycle_max_warmup_cycles = 64;
      constexpr uint32_t lifecycle_min_warmup_cycles = 16;
      constexpr uint32_t lifecycle_stable_points = 8;
      constexpr uint32_t lifecycle_measured_cycles = 128;
      std::vector<LifecycleResourceSnapshot> lifecycle_warm_snapshots;
      lifecycle_warm_snapshots.reserve(lifecycle_max_warmup_cycles);
      LifecycleResourceSnapshot lifecycle_baseline{};
      bool lifecycle_stabilized = false;
      uint32_t lifecycle_warmup_cycles = 0;
      for (uint32_t cycle = 1; cycle <= lifecycle_max_warmup_cycles;
           ++cycle) {
         const LifecycleCycleResult cycle_result =
            lifecycle_cycle("warmup", cycle);
         if (cycle_result == LifecycleCycleResult::fail_safe) {
            cleanup();
            return false;
         }
         const LifecycleResourceSnapshot snapshot =
            lifecycle_resource_snapshot();
         if (!lifecycle_snapshot_complete(snapshot)) {
            lifecycle_log_snapshot("INCONCLUSIVE", "warmup", cycle,
                                   lifecycle_proc_warm_two, snapshot);
            if (lifecycle_dut) {
               cleanup();
               return false;
            }
            break;
         }
         lifecycle_warm_snapshots.push_back(snapshot);
         if (cycle >= lifecycle_min_warmup_cycles &&
             lifecycle_warm_snapshots.size() >= lifecycle_stable_points) {
            const size_t first = lifecycle_warm_snapshots.size() -
                                 lifecycle_stable_points;
            lifecycle_stabilized = true;
            for (size_t i = first + 1;
                 i < lifecycle_warm_snapshots.size(); ++i) {
               lifecycle_stabilized &= lifecycle_ownership_equal(
                  lifecycle_warm_snapshots[first],
                  lifecycle_warm_snapshots[i]);
            }
            if (lifecycle_stabilized) {
               lifecycle_baseline = lifecycle_warm_snapshots.back();
               lifecycle_warmup_cycles = cycle;
               break;
            }
         }
      }

      if (!lifecycle_stabilized) {
         const LifecycleResourceSnapshot current =
            lifecycle_warm_snapshots.empty()
               ? lifecycle_proc_warm_two
               : lifecycle_warm_snapshots.back();
         lifecycle_log_snapshot("INCONCLUSIVE", "warmup",
                                lifecycle_warm_snapshots.size(),
                                lifecycle_proc_warm_two, current);
         if (lifecycle_dut) {
            cleanup();
            return false;
         }
      } else {
         bool lifecycle_resource_drift = false;
         bool lifecycle_sampling_inconclusive = false;
         uint32_t lifecycle_first_drift_cycle = 0;
         uint32_t lifecycle_first_invalid_cycle = 0;
         LifecycleResourceSnapshot lifecycle_first_drift{};
         LifecycleResourceSnapshot lifecycle_first_invalid{};
         LifecycleResourceSnapshot lifecycle_final = lifecycle_baseline;
         long lifecycle_rss_min = lifecycle_baseline.rss_kb;
         long lifecycle_rss_max = lifecycle_baseline.rss_kb;
         for (uint32_t cycle = 1; cycle <= lifecycle_measured_cycles;
              ++cycle) {
            const LifecycleCycleResult cycle_result =
               lifecycle_cycle("measured", cycle);
            if (cycle_result == LifecycleCycleResult::fail_safe) {
               cleanup();
               return false;
            }
            lifecycle_final = lifecycle_resource_snapshot();
            if (!lifecycle_snapshot_complete(lifecycle_final)) {
               lifecycle_log_snapshot("INCONCLUSIVE", "measured", cycle,
                                      lifecycle_baseline, lifecycle_final);
               if (lifecycle_dut) {
                  cleanup();
                  return false;
               }
               if (!lifecycle_sampling_inconclusive) {
                  lifecycle_sampling_inconclusive = true;
                  lifecycle_first_invalid_cycle = cycle;
                  lifecycle_first_invalid = lifecycle_final;
               }
               continue;
            }
            lifecycle_rss_min =
               std::min(lifecycle_rss_min, lifecycle_final.rss_kb);
            lifecycle_rss_max =
               std::max(lifecycle_rss_max, lifecycle_final.rss_kb);
            if (!lifecycle_ownership_equal(lifecycle_baseline,
                                           lifecycle_final) &&
                !lifecycle_resource_drift) {
               lifecycle_resource_drift = true;
               lifecycle_first_drift_cycle = cycle;
               lifecycle_first_drift = lifecycle_final;
            }
         }

         const char *lifecycle_result = "PASS_STABLE";
         if (lifecycle_sampling_inconclusive)
            lifecycle_result = "INCONCLUSIVE";
         else if (lifecycle_resource_drift)
            lifecycle_result = lifecycle_dut ? "FAIL_RESOURCE_DRIFT"
                                             : "CONTROL_DRIFT";
         lifecycle_log_snapshot(lifecycle_result, "measured",
                                lifecycle_sampling_inconclusive
                                   ? lifecycle_first_invalid_cycle
                                   : lifecycle_resource_drift
                                   ? lifecycle_first_drift_cycle
                                   : lifecycle_measured_cycles,
                                lifecycle_baseline,
                                lifecycle_sampling_inconclusive
                                   ? lifecycle_first_invalid
                                   : lifecycle_resource_drift
                                   ? lifecycle_first_drift
                                   : lifecycle_final);
         LOGI("lifecycle-churn[%s]: SUMMARY result=%s role=%s gate=%s "
              "warmup_cycles=%u stable_points=%u measured_cycles=%u "
              "rss_kb_min=%ld rss_kb_max=%ld rss_gate=diagnostic",
              label, lifecycle_result, lifecycle_role, lifecycle_gate,
              lifecycle_warmup_cycles, lifecycle_stable_points,
              lifecycle_measured_cycles, lifecycle_rss_min,
              lifecycle_rss_max);
         if (lifecycle_dut && lifecycle_resource_drift) {
            cleanup();
            return false;
         }
      }
   }

   auto async_submission_probe = [&]() {
      constexpr uint32_t stamp_count = 4096;
      constexpr uint32_t wave_size = 256;
      constexpr uint32_t wave_count = stamp_count / wave_size;
      constexpr uint32_t wave_warmup = 8;
      constexpr uint32_t semaphore_cycles = 64;
      constexpr uint32_t semaphore_interval = 8;
      constexpr uint32_t semaphore_checkpoints =
         semaphore_cycles / semaphore_interval - 1;
      constexpr uint32_t chain_size = 121;
      constexpr uint32_t stamp_sentinel = 0xcdcdcdcdu;
      constexpr uint32_t error_sentinel = 0xffffffffu;
      constexpr uint32_t guard_front = 0x13579bdfu;
      constexpr uint32_t guard_back = 0xfdb97531u;
      constexpr uint32_t waves_terminal = 0x57415645u;
      constexpr uint32_t chain_terminal = 0x43484149u;
      constexpr uint32_t state_words = 4 + stamp_count + 1;
      constexpr VkDeviceSize state_size = state_words * sizeof(uint32_t);
      constexpr uint64_t fence_timeout_ns = 30ull * 1000 * 1000 * 1000;
      VkBuffer state_buffer{};
      VkDeviceMemory state_memory{};
      void *state_map{};
      VkDescriptorSetLayout async_descriptor_layout{};
      VkDescriptorPool async_descriptor_pool{};
      VkShaderModule async_shader{};
      VkPipelineLayout async_pipeline_layout{};
      VkPipeline async_pipeline{};
      VkCommandPool async_pool{};
      VkFence async_fence{};
      VkSemaphore async_semaphore{};
      std::array<VkCommandBuffer, wave_size> commands{};
      bool submitted = false;

      auto debug_has = [](const char *value, const char *wanted) {
         if (!value)
            return false;
         const size_t wanted_length = std::strlen(wanted);
         for (const char *token = value; *token;) {
            while (*token == ',' || *token == ';' || *token == ':' ||
                   *token == ' ' || *token == '\t')
               ++token;
            const char *end = token;
            while (*end && *end != ',' && *end != ';' && *end != ':' &&
                   *end != ' ' && *end != '\t')
               ++end;
            if (static_cast<size_t>(end - token) == wanted_length &&
                std::memcmp(token, wanted, wanted_length) == 0)
               return true;
            token = end;
         }
         return false;
      };
      auto enabled_bool_option = [](const char *value) {
         return value &&
                (!std::strcmp(value, "1") || !strcasecmp(value, "y") ||
                 !strcasecmp(value, "yes") || !strcasecmp(value, "t") ||
                 !strcasecmp(value, "true"));
      };
      if (std::strcmp(label, "panvk") == 0) {
         const bool wait_submit = std::getenv("PANVK_KBASE_WAIT_SUBMIT");
         const bool sync_submit = std::getenv("PANVK_KBASE_SYNC_SUBMIT");
         const bool v9_trace =
            enabled_bool_option(std::getenv("PANVK_V9_TRACE"));
         const bool heap_log = std::getenv("PANVK_HEAP_LOG");
         const char *debug = std::getenv("PANVK_DEBUG");
         const bool debug_all = debug_has(debug, "all");
         const bool debug_trace = debug_all || debug_has(debug, "trace");
         const bool debug_dump = debug_all || debug_has(debug, "dump");
         const bool asynchronous =
            !(wait_submit || sync_submit || v9_trace || heap_log ||
              debug_trace || debug_dump);
         LOGI("jm-async[%s]: MODE wait_submit=%u sync_submit=%u v9_trace=%u "
              "heap_log=%u debug_trace=%u debug_dump=%u asynchronous=%u",
              label, wait_submit, sync_submit, v9_trace, heap_log, debug_trace,
              debug_dump, asynchronous);
         if (!asynchronous) {
            LOGE("jm-async[%s]: NON_COUNTING reason=non-asynchronous-mode",
                 label);
            return false;
         }
      } else {
         LOGI("jm-async[%s]: MODE asynchronous=n/a", label);
      }

      struct ResourceSnapshot {
         long fds = -1;
         long maps = -1;
         long rss_kb = -1;
         long size_kb = -1;
         uint64_t mapped_bytes = UINT64_MAX;
      };
      auto resource_snapshot = []() {
         ResourceSnapshot snapshot;
         if (DIR *directory = opendir("/proc/self/fd")) {
            snapshot.fds = 0;
            while (dirent *entry = readdir(directory)) {
               if (std::strcmp(entry->d_name, ".") &&
                   std::strcmp(entry->d_name, ".."))
                  ++snapshot.fds;
            }
            closedir(directory);
         }
         if (FILE *maps = std::fopen("/proc/self/maps", "r")) {
            snapshot.maps = 0;
            snapshot.mapped_bytes = 0;
            bool valid = true;
            char *line = nullptr;
            size_t capacity = 0;
            while (getline(&line, &capacity, maps) >= 0) {
               unsigned long long start = 0;
               unsigned long long end = 0;
               ++snapshot.maps;
               if (std::sscanf(line, "%llx-%llx", &start, &end) != 2 ||
                   end < start ||
                   UINT64_MAX - snapshot.mapped_bytes < end - start) {
                  valid = false;
                  continue;
               }
               snapshot.mapped_bytes += end - start;
            }
            std::free(line);
            if (std::ferror(maps) || !valid) {
               snapshot.maps = -1;
               snapshot.mapped_bytes = UINT64_MAX;
            }
            std::fclose(maps);
         }
         if (FILE *status = std::fopen("/proc/self/status", "r")) {
            char line[256];
            while (std::fgets(line, sizeof(line), status)) {
               std::sscanf(line, "VmRSS: %ld kB", &snapshot.rss_kb);
               std::sscanf(line, "VmSize: %ld kB", &snapshot.size_kb);
            }
            std::fclose(status);
         }
         return snapshot;
      };
      auto snapshot_complete = [](const ResourceSnapshot &snapshot) {
         return snapshot.fds >= 0 && snapshot.maps >= 0 &&
                snapshot.rss_kb >= 0 && snapshot.size_kb >= 0 &&
                snapshot.mapped_bytes != UINT64_MAX;
      };
      const bool resource_dut = std::strcmp(label, "panvk") == 0;
      const char *resource_record =
         resource_dut ? "RESOURCE_GATE" : "RESOURCE_CONTROL";
      const char *resource_role = resource_dut ? "DUT" : "CONTROL";
      const char *resource_policy = resource_dut ? "DUT" : "DIAGNOSTIC";
      auto require_snapshot = [&](const char *phase, const char *checkpoint,
                                  const ResourceSnapshot &snapshot) {
         if (snapshot_complete(snapshot))
            return true;
         LOGE("jm-async[%s]: %s sampling=NON_COUNTING "
              "result=NON_COUNTING gate=%s role=%s phase=%s "
              "reason=invalid-snapshot "
              "checkpoint=%s fd=%ld maps=%ld mapped_bytes=%llu "
              "size_kb=%ld rss_kb=%ld",
              label, resource_record, resource_policy, resource_role, phase,
              checkpoint, snapshot.fds, snapshot.maps,
              static_cast<unsigned long long>(snapshot.mapped_bytes),
              snapshot.size_kb, snapshot.rss_kb);
         return false;
      };
      auto resource_gate = [&](const char *phase, uint32_t warmup,
                               uint32_t work_count, const char *work_unit,
                               const ResourceSnapshot &baseline,
                               const ResourceSnapshot *checkpoints,
                               size_t checkpoint_count,
                               size_t median_width,
                               uint32_t first_checkpoint,
                               uint32_t checkpoint_stride) {
         if (!snapshot_complete(baseline) || !checkpoints ||
             median_width == 0 || median_width > 4 ||
             checkpoint_count < median_width * 2) {
            LOGE("jm-async[%s]: %s sampling=NON_COUNTING "
                 "result=NON_COUNTING gate=%s role=%s phase=%s "
                 "reason=incomplete-work intervals=%zu median_width=%zu",
                 label, resource_record, resource_policy, resource_role,
                 phase, checkpoint_count, median_width);
            return false;
         }
         for (size_t i = 0; i < checkpoint_count; ++i) {
            if (!snapshot_complete(checkpoints[i])) {
               char checkpoint[24];
               std::snprintf(checkpoint, sizeof(checkpoint), "%u",
                             first_checkpoint +
                                static_cast<uint32_t>(i) * checkpoint_stride);
               return require_snapshot(phase, checkpoint, checkpoints[i]);
            }
         }

         enum class DriftClass {
            stable,
            bounded_drift,
            inconclusive_late_drift,
            growth,
         };
         struct DriftSummary {
            DriftClass classification = DriftClass::stable;
            size_t first_change_index = SIZE_MAX;
            uint32_t transitions = 0;
            size_t plateau_points = 0;
         };
         auto classify = [&](auto equal) {
            DriftSummary summary;
            size_t last_change_index = SIZE_MAX;
            const ResourceSnapshot *previous = &baseline;
            for (size_t i = 0; i < checkpoint_count; ++i) {
               if (!equal(*previous, checkpoints[i])) {
                  if (summary.first_change_index == SIZE_MAX)
                     summary.first_change_index = i;
                  last_change_index = i;
                  ++summary.transitions;
               }
               previous = &checkpoints[i];
            }
            summary.plateau_points = summary.transitions == 0
                                        ? checkpoint_count
                                        : checkpoint_count -
                                             last_change_index - 1;
            if (summary.transitions == 0) {
               summary.classification = DriftClass::stable;
            } else if (summary.transitions == 1 &&
                       summary.plateau_points >= 4) {
               summary.classification = DriftClass::bounded_drift;
            } else if (summary.transitions == 1) {
               summary.classification =
                  DriftClass::inconclusive_late_drift;
            } else {
               summary.classification = DriftClass::growth;
            }
            return summary;
         };
         auto class_name = [](DriftClass classification) {
            switch (classification) {
            case DriftClass::stable:
               return "STABLE";
            case DriftClass::bounded_drift:
               return "BOUNDED_DRIFT";
            case DriftClass::inconclusive_late_drift:
               return "INCONCLUSIVE_LATE_DRIFT";
            case DriftClass::growth:
               return "GROWTH";
            }
            return "INCONCLUSIVE_LATE_DRIFT";
         };
         auto first_change_checkpoint = [&](const DriftSummary &summary) {
            return summary.first_change_index == SIZE_MAX
                      ? 0u
                      : first_checkpoint +
                           static_cast<uint32_t>(summary.first_change_index) *
                              checkpoint_stride;
         };
         auto median = [&](size_t first, bool size_metric) {
            std::array<long, 4> values{};
            for (size_t i = 0; i < median_width; ++i) {
               const ResourceSnapshot &snapshot = checkpoints[first + i];
               values[i] = size_metric ? snapshot.size_kb : snapshot.rss_kb;
            }
            std::sort(values.begin(), values.begin() + median_width);
            return median_width & 1
                      ? values[median_width / 2]
                      : (values[median_width / 2 - 1] +
                         values[median_width / 2]) /
                           2;
         };

         long fd_max = baseline.fds;
         long maps_max = baseline.maps;
         uint64_t mapped_bytes_max = baseline.mapped_bytes;
         long size_min = baseline.size_kb;
         long size_max = baseline.size_kb;
         long rss_min = baseline.rss_kb;
         long rss_max = baseline.rss_kb;
         bool fd_stable = true;
         bool maps_stable = true;
         bool mapped_bytes_stable = true;
         uint32_t size_positive_transitions = 0;
         uint32_t size_positive_run = 0;
         uint32_t size_max_positive_run = 0;
         long previous_size = baseline.size_kb;
         for (size_t i = 0; i < checkpoint_count; ++i) {
            const ResourceSnapshot &snapshot = checkpoints[i];
            fd_stable &= snapshot.fds == baseline.fds;
            maps_stable &= snapshot.maps == baseline.maps;
            mapped_bytes_stable &=
               snapshot.mapped_bytes == baseline.mapped_bytes;
            fd_max = std::max(fd_max, snapshot.fds);
            maps_max = std::max(maps_max, snapshot.maps);
            mapped_bytes_max =
               std::max(mapped_bytes_max, snapshot.mapped_bytes);
            size_min = std::min(size_min, snapshot.size_kb);
            size_max = std::max(size_max, snapshot.size_kb);
            rss_min = std::min(rss_min, snapshot.rss_kb);
            rss_max = std::max(rss_max, snapshot.rss_kb);

            if (snapshot.size_kb > previous_size)
               ++size_positive_transitions;
            size_positive_run = snapshot.size_kb > previous_size
                                   ? size_positive_run + 1
                                   : 0;
            size_max_positive_run =
               std::max(size_max_positive_run, size_positive_run);
            previous_size = snapshot.size_kb;
         }

         const long first_size_median = median(0, true);
         const long last_size_median =
            median(checkpoint_count - median_width, true);
         const long first_rss_median = median(0, false);
         const long last_rss_median =
            median(checkpoint_count - median_width, false);
         const bool size_sustained_growth =
            last_size_median - first_size_median > 1024 &&
            size_max_positive_run >= 3;
         const DriftSummary ownership = classify(
            [](const ResourceSnapshot &left, const ResourceSnapshot &right) {
               return left.fds == right.fds && left.maps == right.maps &&
                      left.mapped_bytes == right.mapped_bytes;
            });
         const DriftSummary size = classify(
            [](const ResourceSnapshot &left, const ResourceSnapshot &right) {
               return left.size_kb == right.size_kb;
            });
         const DriftClass observed =
            static_cast<unsigned>(ownership.classification) >=
                  static_cast<unsigned>(size.classification)
               ? ownership.classification
               : size.classification;
         const ResourceSnapshot &final = checkpoints[checkpoint_count - 1];
         const bool ownership_drift = ownership.transitions != 0;
         const bool size_net_upward =
            final.size_kb > baseline.size_kb ||
            last_size_median > first_size_median;
         const bool size_growth =
            size.classification == DriftClass::growth &&
            ((size_positive_transitions >= 2 && size_net_upward) ||
             size_sustained_growth);
         bool accepted = true;
         const char *dut_result = "PASS_STABLE";
         if (ownership_drift) {
            dut_result = "FAIL_OWNERSHIP_DRIFT";
            accepted = false;
         } else {
            switch (size.classification) {
            case DriftClass::stable:
               dut_result = "PASS_STABLE";
               break;
            case DriftClass::bounded_drift:
               dut_result = "PASS_BOUNDED_DRIFT";
               break;
            case DriftClass::inconclusive_late_drift:
               dut_result = "INCONCLUSIVE_LATE_DRIFT";
               accepted = false;
               break;
            case DriftClass::growth:
               dut_result = size_growth ? "FAIL_GROWTH" : "PASS_NO_GROWTH";
               accepted = !size_growth;
               break;
            }
         }
         const char *reason = "none";
         if (!fd_stable)
            reason = "fd-drift";
         else if (!maps_stable)
            reason = "map-count-drift";
         else if (!mapped_bytes_stable)
            reason = "mapped-bytes-drift";
         else if (size.classification ==
                  DriftClass::inconclusive_late_drift)
            reason = "vmsize-late-drift";
         else if (size_growth)
            reason = "vmsize-growth";
         else if (size.classification == DriftClass::bounded_drift)
            reason = "bounded-vmsize-drift";
         else if (size.classification == DriftClass::growth)
            reason = "no-net-vmsize-growth";
         const char *result =
            resource_dut ? dut_result : class_name(observed);
         __android_log_print(
            resource_dut && !accepted ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO,
            "PanvkPresent",
            "jm-async[%s]: %s sampling=VALID result=%s gate=%s role=%s "
            "phase=%s "
            "heuristic=target-probe-vmsize "
            "warmup=%u intervals=%zu work=%u work_unit=%s "
            "ownership_class=%s ownership_first_change_checkpoint=%u "
            "ownership_transitions=%u ownership_plateau_points=%zu "
            "size_class=%s size_first_change_checkpoint=%u "
            "size_transitions=%u "
            "size_plateau_points=%zu "
            "fd_baseline=%ld fd_max=%ld fd_final=%ld "
            "maps_baseline=%ld maps_max=%ld maps_final=%ld "
            "mapped_bytes_baseline=%llu mapped_bytes_max=%llu "
            "mapped_bytes_final=%llu "
            "size_kb_baseline=%ld size_kb_min=%ld size_kb_max=%ld "
            "size_kb_final=%ld size_median=%ld->%ld "
            "size_positive_transitions=%u size_positive_run=%u "
            "rss_kb_baseline=%ld rss_kb_min=%ld rss_kb_max=%ld "
            "rss_kb_final=%ld rss_median=%ld->%ld rss_gate=diagnostic "
            "reason=%s scope=same-process-target-probe",
            label, resource_record, result, resource_policy, resource_role,
            phase, warmup, checkpoint_count, work_count, work_unit,
            class_name(ownership.classification),
            first_change_checkpoint(ownership), ownership.transitions,
            ownership.plateau_points, class_name(size.classification),
            first_change_checkpoint(size), size.transitions,
            size.plateau_points, baseline.fds, fd_max, final.fds,
            baseline.maps, maps_max, final.maps,
            static_cast<unsigned long long>(baseline.mapped_bytes),
            static_cast<unsigned long long>(mapped_bytes_max),
            static_cast<unsigned long long>(final.mapped_bytes),
            baseline.size_kb, size_min, size_max, final.size_kb,
            first_size_median, last_size_median, size_positive_transitions,
            size_max_positive_run, baseline.rss_kb, rss_min, rss_max,
            final.rss_kb, first_rss_median, last_rss_median, reason);
         return !resource_dut || accepted;
      };

      auto async_cleanup = [&]() {
         if (state_map)
            UnmapMemory(device, state_memory);
         if (async_fence)
            DestroyFence(device, async_fence, nullptr);
         if (async_semaphore)
            DestroySemaphore(device, async_semaphore, nullptr);
         if (async_pool)
            DestroyCommandPool(device, async_pool, nullptr);
         if (async_pipeline)
            DestroyPipeline(device, async_pipeline, nullptr);
         if (async_pipeline_layout)
            DestroyPipelineLayout(device, async_pipeline_layout, nullptr);
         if (async_shader)
            DestroyShaderModule(device, async_shader, nullptr);
         if (async_descriptor_pool)
            DestroyDescriptorPool(device, async_descriptor_pool, nullptr);
         if (async_descriptor_layout)
            DestroyDescriptorSetLayout(device, async_descriptor_layout,
                                       nullptr);
         if (state_buffer)
            DestroyBuffer(device, state_buffer, nullptr);
         if (state_memory)
            FreeMemory(device, state_memory, nullptr);
      };
      auto async_failed = [&](const char *operation, VkResult failure) {
         LOGE("jm-async[%s]: FAIL operation=%s result=%d submitted=%u", label,
              operation, failure, submitted ? 1u : 0u);
         if (submitted)
            _exit(124);
         async_cleanup();
         return false;
      };

      VkBufferCreateInfo async_buffer_info{
         VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      async_buffer_info.size = state_size;
      async_buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      async_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VkResult async_result = CreateBuffer(device, &async_buffer_info, nullptr,
                                           &state_buffer);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateBuffer", async_result);
      VkMemoryRequirements async_requirements{};
      GetBufferMemoryRequirements(device, state_buffer, &async_requirements);
      uint32_t async_memory_type = UINT32_MAX;
      for (uint32_t pass = 0; pass < 2 && async_memory_type == UINT32_MAX;
           ++pass) {
         for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags flags =
               memory_properties.memoryTypes[i].propertyFlags;
            if ((async_requirements.memoryTypeBits & (1u << i)) &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (pass || !(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
               async_memory_type = i;
               break;
            }
         }
      }
      if (async_memory_type == UINT32_MAX) {
         LOGE("jm-async[%s]: FAIL reason=no-host-visible-memory", label);
         async_cleanup();
         return false;
      }
      const VkMemoryPropertyFlags async_memory_flags =
         memory_properties.memoryTypes[async_memory_type].propertyFlags;
      VkMemoryAllocateInfo async_alloc_info{
         VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      async_alloc_info.allocationSize = async_requirements.size;
      async_alloc_info.memoryTypeIndex = async_memory_type;
      async_result = AllocateMemory(device, &async_alloc_info, nullptr,
                                    &state_memory);
      if (async_result != VK_SUCCESS)
         return async_failed("vkAllocateMemory", async_result);
      async_result = BindBufferMemory(device, state_buffer, state_memory, 0);
      if (async_result != VK_SUCCESS)
         return async_failed("vkBindBufferMemory", async_result);
      async_result = MapMemory(device, state_memory, 0, VK_WHOLE_SIZE, 0,
                               &state_map);
      if (async_result != VK_SUCCESS)
         return async_failed("vkMapMemory", async_result);

      VkDescriptorSetLayoutBinding async_binding{};
      async_binding.binding = 0;
      async_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      async_binding.descriptorCount = 1;
      async_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      VkDescriptorSetLayoutCreateInfo async_layout_info{
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      async_layout_info.bindingCount = 1;
      async_layout_info.pBindings = &async_binding;
      async_result = CreateDescriptorSetLayout(
         device, &async_layout_info, nullptr, &async_descriptor_layout);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateDescriptorSetLayout", async_result);
      VkDescriptorPoolSize async_pool_size{
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
      VkDescriptorPoolCreateInfo async_descriptor_pool_info{
         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      async_descriptor_pool_info.maxSets = 1;
      async_descriptor_pool_info.poolSizeCount = 1;
      async_descriptor_pool_info.pPoolSizes = &async_pool_size;
      async_result = CreateDescriptorPool(device, &async_descriptor_pool_info,
                                          nullptr, &async_descriptor_pool);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateDescriptorPool", async_result);
      VkDescriptorSet async_descriptor_set{};
      VkDescriptorSetAllocateInfo async_descriptor_allocate_info{
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      async_descriptor_allocate_info.descriptorPool = async_descriptor_pool;
      async_descriptor_allocate_info.descriptorSetCount = 1;
      async_descriptor_allocate_info.pSetLayouts = &async_descriptor_layout;
      async_result = AllocateDescriptorSets(
         device, &async_descriptor_allocate_info, &async_descriptor_set);
      if (async_result != VK_SUCCESS)
         return async_failed("vkAllocateDescriptorSets", async_result);
      VkDescriptorBufferInfo async_descriptor_buffer{state_buffer, 0,
                                                     state_size};
      VkWriteDescriptorSet async_descriptor_write{
         VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      async_descriptor_write.dstSet = async_descriptor_set;
      async_descriptor_write.dstBinding = 0;
      async_descriptor_write.descriptorCount = 1;
      async_descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      async_descriptor_write.pBufferInfo = &async_descriptor_buffer;
      UpdateDescriptorSets(device, 1, &async_descriptor_write, 0, nullptr);

      VkShaderModuleCreateInfo async_shader_info{
         VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      async_shader_info.codeSize = sizeof(panvk_async_spv_bytes);
      async_shader_info.pCode = reinterpret_cast<const uint32_t *>(
         panvk_async_spv_bytes);
      async_result = CreateShaderModule(device, &async_shader_info, nullptr,
                                        &async_shader);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateShaderModule", async_result);
      VkPushConstantRange async_push_range{};
      async_push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      async_push_range.size = 3 * sizeof(uint32_t);
      VkPipelineLayoutCreateInfo async_pipeline_layout_info{
         VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      async_pipeline_layout_info.setLayoutCount = 1;
      async_pipeline_layout_info.pSetLayouts = &async_descriptor_layout;
      async_pipeline_layout_info.pushConstantRangeCount = 1;
      async_pipeline_layout_info.pPushConstantRanges = &async_push_range;
      async_result = CreatePipelineLayout(device, &async_pipeline_layout_info,
                                          nullptr, &async_pipeline_layout);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreatePipelineLayout", async_result);
      VkPipelineShaderStageCreateInfo async_stage{
         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      async_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      async_stage.module = async_shader;
      async_stage.pName = "main";
      VkComputePipelineCreateInfo async_pipeline_info{
         VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      async_pipeline_info.stage = async_stage;
      async_pipeline_info.layout = async_pipeline_layout;
      async_result = CreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                            &async_pipeline_info, nullptr,
                                            &async_pipeline);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateComputePipelines", async_result);

      VkCommandPoolCreateInfo async_command_pool_info{
         VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      async_command_pool_info.flags =
         VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      async_command_pool_info.queueFamilyIndex = queue_family;
      async_result = CreateCommandPool(device, &async_command_pool_info,
                                       nullptr, &async_pool);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateCommandPool", async_result);
      VkCommandBufferAllocateInfo async_command_allocate_info{
         VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      async_command_allocate_info.commandPool = async_pool;
      async_command_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      async_command_allocate_info.commandBufferCount = commands.size();
      async_result = AllocateCommandBuffers(device,
                                            &async_command_allocate_info,
                                            commands.data());
      if (async_result != VK_SUCCESS)
         return async_failed("vkAllocateCommandBuffers", async_result);
      VkFenceCreateInfo async_fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      async_result = CreateFence(device, &async_fence_info, nullptr,
                                 &async_fence);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateFence", async_result);
      VkSemaphoreCreateInfo async_semaphore_info{
         VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      async_result = CreateSemaphore(device, &async_semaphore_info, nullptr,
                                     &async_semaphore);
      if (async_result != VK_SUCCESS)
         return async_failed("vkCreateSemaphore", async_result);

      auto initialize_state = [&]() {
         auto *words = static_cast<uint32_t *>(state_map);
         words[0] = guard_front;
         words[1] = 0;
         words[2] = error_sentinel;
         words[3] = stamp_sentinel;
         std::fill(words + 4, words + 4 + stamp_count, stamp_sentinel);
         words[state_words - 1] = guard_back;
         if (async_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            return VK_SUCCESS;
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = state_memory;
         range.size = VK_WHOLE_SIZE;
         return FlushMappedMemoryRanges(device, 1, &range);
      };
      auto invalidate_state = [&]() {
         if (async_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            return VK_SUCCESS;
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = state_memory;
         range.size = VK_WHOLE_SIZE;
         return InvalidateMappedMemoryRanges(device, 1, &range);
      };
      auto verify_state = [&](uint32_t completed, uint32_t terminal,
                              const char *phase) {
         const auto *words = static_cast<const uint32_t *>(state_map);
         if (words[0] != guard_front || words[state_words - 1] != guard_back ||
             words[1] != completed || words[2] != error_sentinel ||
             words[3] != terminal) {
            LOGE("jm-async[%s]: FAIL phase=%s counter=%u error=%08x "
                 "guard_front=%08x sentinel=%08x guard_back=%08x",
                 label, phase, words[1], words[2], words[0], words[3],
                 words[state_words - 1]);
            return false;
         }
         for (uint32_t i = 0; i < stamp_count; ++i) {
            const uint32_t expected =
               i < completed ? (i * 0x9e3779b9u) ^ 0xa5a55a5au
                             : stamp_sentinel;
            if (words[4 + i] != expected) {
               LOGE("jm-async[%s]: FAIL phase=%s stamp=%u expected=%08x "
                    "actual=%08x",
                    label, phase, i, expected, words[4 + i]);
               return false;
            }
         }
         return true;
      };
      auto record = [&](VkCommandBuffer command_buffer, uint32_t expected,
                        uint32_t terminal, bool host_written,
                        bool host_read) {
         VkResult command_result = ResetCommandBuffer(command_buffer, 0);
         if (command_result != VK_SUCCESS)
            return command_result;
         VkCommandBufferBeginInfo async_begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
         async_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
         command_result = BeginCommandBuffer(command_buffer, &async_begin);
         if (command_result != VK_SUCCESS)
            return command_result;
         if (host_written) {
            VkBufferMemoryBarrier host_to_compute{
               VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_to_compute.dstAccessMask =
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            host_to_compute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_to_compute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_to_compute.buffer = state_buffer;
            host_to_compute.size = state_size;
            CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                               nullptr, 1, &host_to_compute, 0, nullptr);
         }
         CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         async_pipeline);
         CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               async_pipeline_layout, 0, 1,
                               &async_descriptor_set, 0, nullptr);
         const uint32_t push[3] = {expected, terminal ? 1u : 0u, terminal};
         CmdPushConstants(command_buffer, async_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
         CmdDispatch(command_buffer, 1, 1, 1);
         VkBufferMemoryBarrier compute_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
         compute_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
         compute_barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            (host_read ? VK_ACCESS_HOST_READ_BIT : 0);
         compute_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         compute_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         compute_barrier.buffer = state_buffer;
         compute_barrier.size = state_size;
         const VkPipelineStageFlags destination_stages =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            (host_read ? VK_PIPELINE_STAGE_HOST_BIT : 0);
         CmdPipelineBarrier(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            destination_stages, 0, 0, nullptr, 1,
                            &compute_barrier, 0, nullptr);
         return EndCommandBuffer(command_buffer);
      };
      uint32_t immediate_fences = 0;
      auto wait_for_batch = [&](const char *phase) {
         VkResult status = GetFenceStatus(device, async_fence);
         if (status == VK_SUCCESS)
            ++immediate_fences;
         else if (status != VK_NOT_READY)
            return async_failed("vkGetFenceStatus", status);
         VkResult wait_result = WaitForFences(device, 1, &async_fence,
                                              VK_TRUE, fence_timeout_ns);
         if (wait_result != VK_SUCCESS)
            return async_failed(phase, wait_result);
         return true;
      };

      const ResourceSnapshot wave_throwaway = resource_snapshot();
      if (!require_snapshot("waves", "throwaway", wave_throwaway)) {
         async_cleanup();
         return false;
      }
      std::array<ResourceSnapshot, wave_count> wave_resources{};
      async_result = initialize_state();
      if (async_result != VK_SUCCESS)
         return async_failed("vkFlushMappedMemoryRanges(waves)", async_result);
      for (uint32_t wave = 0; wave < wave_count; ++wave) {
         for (uint32_t i = 0; i < wave_size; ++i) {
            const uint32_t expected = wave * wave_size + i;
            const uint32_t terminal =
               expected + 1 == stamp_count ? waves_terminal : 0;
            async_result = record(commands[i], expected, terminal,
                                  expected == 0,
                                  i + 1 == wave_size);
            if (async_result != VK_SUCCESS)
               return async_failed("record(waves)", async_result);
         }
         for (uint32_t i = 0; i < wave_size; ++i) {
            VkSubmitInfo async_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            async_submit.commandBufferCount = 1;
            async_submit.pCommandBuffers = &commands[i];
            async_result = QueueSubmit(queue, 1, &async_submit,
                                       i + 1 == wave_size ? async_fence
                                                          : VK_NULL_HANDLE);
            if (async_result != VK_SUCCESS)
               return async_failed("vkQueueSubmit(waves)", async_result);
            submitted = true;
         }
         if (!wait_for_batch("vkWaitForFences(waves)"))
            return false;
         submitted = false;
         async_result = invalidate_state();
         if (async_result != VK_SUCCESS)
            return async_failed("vkInvalidateMappedMemoryRanges(waves)",
                                async_result);
         const uint32_t expected_terminal =
            wave + 1 == wave_count ? waves_terminal : stamp_sentinel;
         if (!verify_state((wave + 1) * wave_size, expected_terminal,
                           "waves")) {
            async_cleanup();
            return false;
         }
         wave_resources[wave] = resource_snapshot();
         char checkpoint[24];
         std::snprintf(checkpoint, sizeof(checkpoint), "wave-%u", wave + 1);
         if (!require_snapshot("waves", checkpoint, wave_resources[wave])) {
            async_cleanup();
            return false;
         }
         if (wave + 1 != wave_count) {
            async_result = ResetFences(device, 1, &async_fence);
            if (async_result != VK_SUCCESS)
               return async_failed("vkResetFences(waves)", async_result);
         }
      }
      if (!resource_gate(
             "waves", wave_warmup,
             (wave_count - wave_warmup) * wave_size, "submits",
             wave_resources[wave_warmup - 1],
             wave_resources.data() + wave_warmup,
             wave_count - wave_warmup, 4, wave_warmup + 1, 1)) {
         async_cleanup();
         return false;
      }
      LOGI("jm-async[%s]: PASS mode=waves submissions=%u waves=%u live_cb=%u "
           "terminal=%08x immediate_fences=%u",
           label, stamp_count, wave_count, wave_size, waves_terminal,
           immediate_fences);

      async_result = ResetFences(device, 1, &async_fence);
      if (async_result != VK_SUCCESS)
         return async_failed("vkResetFences(chain)", async_result);
      async_result = initialize_state();
      if (async_result != VK_SUCCESS)
         return async_failed("vkFlushMappedMemoryRanges(chain)", async_result);
      for (uint32_t i = 0; i < chain_size; ++i) {
         async_result = record(commands[i], i,
                               i + 1 == chain_size ? chain_terminal : 0,
                               i == 0,
                               i + 1 == chain_size);
         if (async_result != VK_SUCCESS)
            return async_failed("record(chain)", async_result);
      }
      VkSubmitInfo chain_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      chain_submit.commandBufferCount = chain_size;
      chain_submit.pCommandBuffers = commands.data();
      async_result = QueueSubmit(queue, 1, &chain_submit, async_fence);
      if (async_result != VK_SUCCESS)
         return async_failed("vkQueueSubmit(chain)", async_result);
      submitted = true;
      if (!wait_for_batch("vkWaitForFences(chain)"))
         return false;
      submitted = false;
      async_result = invalidate_state();
      if (async_result != VK_SUCCESS)
         return async_failed("vkInvalidateMappedMemoryRanges(chain)",
                             async_result);
      if (!verify_state(chain_size, chain_terminal, "chain")) {
         async_cleanup();
         return false;
      }
      LOGI("jm-async[%s]: PASS mode=single-submit-chain command_buffers=%u "
           "terminal=%08x immediate_fences=%u",
           label, chain_size, chain_terminal, immediate_fences);

      ResourceSnapshot semaphore_baseline{};
      std::array<ResourceSnapshot, semaphore_checkpoints>
         semaphore_resources{};
      for (uint32_t cycle = 0; cycle < semaphore_cycles; ++cycle) {
         const uint32_t terminal = 0x53454d00u + cycle + 1;
         async_result = initialize_state();
         if (async_result != VK_SUCCESS)
            return async_failed("vkFlushMappedMemoryRanges(semaphore)",
                                async_result);
         async_result = ResetFences(device, 1, &async_fence);
         if (async_result != VK_SUCCESS)
            return async_failed("vkResetFences(semaphore)", async_result);
         async_result = record(commands[0], 0, 0, true, false);
         if (async_result != VK_SUCCESS)
            return async_failed("record(semaphore-signal)", async_result);
         async_result = record(commands[1], 1, terminal, false, true);
         if (async_result != VK_SUCCESS)
            return async_failed("record(semaphore-wait)", async_result);

         VkSubmitInfo signal_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
         signal_submit.commandBufferCount = 1;
         signal_submit.pCommandBuffers = &commands[0];
         signal_submit.signalSemaphoreCount = 1;
         signal_submit.pSignalSemaphores = &async_semaphore;
         async_result = QueueSubmit(queue, 1, &signal_submit, VK_NULL_HANDLE);
         if (async_result != VK_SUCCESS)
            return async_failed("vkQueueSubmit(semaphore-signal)", async_result);
         submitted = true;

         const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
         VkSubmitInfo wait_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
         wait_submit.waitSemaphoreCount = 1;
         wait_submit.pWaitSemaphores = &async_semaphore;
         wait_submit.pWaitDstStageMask = &wait_stage;
         wait_submit.commandBufferCount = 1;
         wait_submit.pCommandBuffers = &commands[1];
         async_result = QueueSubmit(queue, 1, &wait_submit, async_fence);
         if (async_result != VK_SUCCESS)
            return async_failed("vkQueueSubmit(semaphore-wait)", async_result);

         if (cycle == 0) {
            std::array<VkResult, 2> wait_results{};
            std::thread waiters[2];
            for (uint32_t i = 0; i < 2; ++i) {
               waiters[i] = std::thread([&, i]() {
                  wait_results[i] = WaitForFences(
                     device, 1, &async_fence, VK_TRUE, fence_timeout_ns);
               });
            }
            for (auto &waiter : waiters)
               waiter.join();
            if (wait_results[0] != VK_SUCCESS)
               return async_failed("vkWaitForFences(concurrent-0)",
                                   wait_results[0]);
            if (wait_results[1] != VK_SUCCESS)
               return async_failed("vkWaitForFences(concurrent-1)",
                                   wait_results[1]);
         } else if (!wait_for_batch("vkWaitForFences(semaphore)")) {
            return false;
         }
         submitted = false;
         async_result = invalidate_state();
         if (async_result != VK_SUCCESS)
            return async_failed("vkInvalidateMappedMemoryRanges(semaphore)",
                                async_result);
         if (!verify_state(2, terminal, "semaphore")) {
            async_cleanup();
            return false;
         }
         const uint32_t completed_cycles = cycle + 1;
         if (completed_cycles % semaphore_interval == 0) {
            const ResourceSnapshot snapshot = resource_snapshot();
            char checkpoint[32];
            std::snprintf(checkpoint, sizeof(checkpoint), "cycle-%u",
                          completed_cycles);
            if (!require_snapshot("binary-semaphore", checkpoint, snapshot)) {
               async_cleanup();
               return false;
            }
            if (completed_cycles == semaphore_interval) {
               semaphore_baseline = snapshot;
            } else {
               semaphore_resources[completed_cycles / semaphore_interval - 2] =
                  snapshot;
            }
         }
      }
      if (!resource_gate("binary-semaphore", semaphore_interval,
                         (semaphore_cycles - semaphore_interval) * 2,
                         "submits", semaphore_baseline,
                         semaphore_resources.data(),
                         semaphore_resources.size(), 3,
                         semaphore_interval * 2, semaphore_interval)) {
         async_cleanup();
         return false;
      }
      LOGI("jm-async[%s]: PASS mode=binary-semaphore "
           "scope=same-queue-reuse cycles=%u concurrent_waiters=2",
           label, semaphore_cycles);
      async_cleanup();
      const ResourceSnapshot post_cleanup = resource_snapshot();
      LOGI("jm-async[%s]: RESOURCE DIAGNOSTIC phase=post-cleanup "
           "available=%u fd=%ld maps=%ld mapped_bytes=%llu "
           "size_kb=%ld rss_kb=%ld",
           label, static_cast<unsigned>(snapshot_complete(post_cleanup)),
           post_cleanup.fds, post_cleanup.maps,
           static_cast<unsigned long long>(post_cleanup.mapped_bytes),
           post_cleanup.size_kb, post_cleanup.rss_kb);
      return true;
   };
   if (!async_submission_probe()) {
      cleanup();
      return false;
   }

   constexpr uint32_t image_width = 31;
   constexpr uint32_t image_height = 29;
   constexpr uint32_t image_layers = 2;
   constexpr uint32_t buffer_row_length = 32;
   constexpr uint32_t buffer_image_height = 30;
   constexpr VkDeviceSize row_pitch = buffer_row_length * 4;
   constexpr VkDeviceSize layer_pitch = row_pitch * buffer_image_height;
   constexpr VkDeviceSize image_buffer_size = layer_pitch * image_layers;
   static_assert(row_pitch == 128, "unexpected image row pitch");
   static_assert(layer_pitch == 3840, "unexpected image layer pitch");
   static_assert(image_buffer_size == 7680,
                 "unexpected image buffer size");
   static_assert(image_buffer_size <= buffer_size,
                 "probe buffers must hold the image layout");
   constexpr VkFormat image_format = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageFormatProperties sampled_properties{};
   result = GetPhysicalDeviceImageFormatProperties(
      physical, image_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
         VK_IMAGE_USAGE_SAMPLED_BIT,
      0, &sampled_properties);
   if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) {
      LOGI("image-transfer[%s]: SKIP unsupported sampled RGBA8 image", label);
      cleanup();
      return false;
   }
   if (result != VK_SUCCESS)
      return failed("image-transfer vkGetPhysicalDeviceImageFormatProperties",
                    result);
   VkImageFormatProperties output_properties{};
   result = GetPhysicalDeviceImageFormatProperties(
      physical, image_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      0, &output_properties);
   if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) {
      LOGI("sample[%s]: SKIP unsupported color-attachment RGBA8 image", label);
      cleanup();
      return false;
   }
   if (result != VK_SUCCESS)
      return failed("sample vkGetPhysicalDeviceImageFormatProperties", result);
   if (sampled_properties.maxExtent.width < image_width ||
       sampled_properties.maxExtent.height < image_height ||
       sampled_properties.maxArrayLayers < image_layers ||
       !(sampled_properties.sampleCounts & VK_SAMPLE_COUNT_1_BIT) ||
       output_properties.maxExtent.width < image_width ||
       output_properties.maxExtent.height < image_height ||
       !(output_properties.sampleCounts & VK_SAMPLE_COUNT_1_BIT)) {
      LOGI("image-transfer[%s]: SKIP unsupported 31x29x2 RGBA8 image", label);
      cleanup();
      return false;
   }

   constexpr uint8_t upload_sentinel = 0xa6;
   constexpr uint8_t transfer_sentinel = 0x5a;
   std::array<uint8_t, image_buffer_size> upload{};
   std::array<uint8_t, image_buffer_size> transfer_expected{};
   upload.fill(upload_sentinel);
   transfer_expected.fill(transfer_sentinel);
   for (uint32_t layer = 0; layer < image_layers; ++layer) {
      for (uint32_t y = 0; y < image_height; ++y) {
         for (uint32_t x = 0; x < image_width; ++x) {
            const size_t offset = layer * layer_pitch + y * row_pitch + x * 4;
            const uint8_t pixel[4] = {
               static_cast<uint8_t>(x < 11 ? 255 : 0),
               static_cast<uint8_t>(y < 7 ? 255 : 0),
               static_cast<uint8_t>(layer == 1 ? 255 : 0),
               static_cast<uint8_t>(x == 23 && y == 17 ? 255 : 0),
            };
            std::memcpy(upload.data() + offset, pixel, sizeof(pixel));
            std::memcpy(transfer_expected.data() + offset, pixel,
                        sizeof(pixel));
         }
      }
   }
   std::memcpy(src_map, upload.data(), image_buffer_size);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = src_memory;
      range.size = VK_WHOLE_SIZE;
      result = FlushMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("image-transfer vkFlushMappedMemoryRanges", result);
   }

   VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
   image_info.imageType = VK_IMAGE_TYPE_2D;
   image_info.format = image_format;
   image_info.extent = {image_width, image_height, 1};
   image_info.mipLevels = 1;
   image_info.arrayLayers = image_layers;
   image_info.samples = VK_SAMPLE_COUNT_1_BIT;
   image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
   image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT;
   image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
   image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   result = CreateImage(device, &image_info, nullptr, &sampled_image);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkCreateImage", result);
   image_info.arrayLayers = 1;
   image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   result = CreateImage(device, &image_info, nullptr, &output_image);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateImage", result);

   auto allocate_image_memory = [&](VkImage image, VkDeviceMemory *memory,
                                    const char *operation) {
      VkMemoryRequirements requirements{};
      GetImageMemoryRequirements(device, image, &requirements);
      uint32_t type = UINT32_MAX;
      for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
         if ((requirements.memoryTypeBits & (1u << i)) &&
             (memory_properties.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
         }
      }
      if (type == UINT32_MAX) {
         for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if (requirements.memoryTypeBits & (1u << i)) {
               type = i;
               break;
            }
         }
      }
      if (type == UINT32_MAX) {
         LOGE("%s[%s]: no compatible image memory type", operation, label);
         return false;
      }
      VkMemoryAllocateInfo image_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      image_alloc.allocationSize = requirements.size;
      image_alloc.memoryTypeIndex = type;
      VkResult image_result =
         AllocateMemory(device, &image_alloc, nullptr, memory);
      if (image_result != VK_SUCCESS) {
         LOGE("%s[%s]: vkAllocateMemory failed: %d", operation, label,
              image_result);
         return false;
      }
      image_result = BindImageMemory(device, image, *memory, 0);
      if (image_result != VK_SUCCESS) {
         LOGE("%s[%s]: vkBindImageMemory failed: %d", operation, label,
              image_result);
         return false;
      }
      return true;
   };
   if (!allocate_image_memory(sampled_image, &sampled_image_memory,
                              "image-transfer") ||
       !allocate_image_memory(output_image, &output_image_memory, "sample")) {
      cleanup();
      return false;
   }

   VkCommandBuffer image_command{};
   result = AllocateCommandBuffers(device, &command_info, &image_command);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkAllocateCommandBuffers", result);
   std::memset(dst_map, transfer_sentinel, image_buffer_size);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = FlushMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("image-transfer dst vkFlushMappedMemoryRanges", result);
   }
   result = BeginCommandBuffer(image_command, &begin_info);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkBeginCommandBuffer", result);
   VkImageMemoryBarrier image_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
   image_barrier.srcAccessMask = 0;
   image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   image_barrier.image = sampled_image;
   image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   image_barrier.subresourceRange.levelCount = 1;
   image_barrier.subresourceRange.layerCount = image_layers;
   CmdPipelineBarrier(image_command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                      nullptr, 1, &image_barrier);
   VkBufferImageCopy image_copy{};
   image_copy.bufferRowLength = buffer_row_length;
   image_copy.bufferImageHeight = buffer_image_height;
   image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   image_copy.imageSubresource.baseArrayLayer = 0;
   image_copy.imageSubresource.layerCount = image_layers;
   image_copy.imageExtent = {image_width, image_height, 1};
   CmdCopyBufferToImage(image_command, src, sampled_image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);
   image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   CmdPipelineBarrier(image_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                      nullptr, 1, &image_barrier);
   CmdCopyImageToBuffer(image_command, sampled_image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1,
                        &image_copy);
   VkBufferMemoryBarrier transfer_host_barrier{
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
   transfer_host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   transfer_host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
   transfer_host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   transfer_host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   transfer_host_barrier.buffer = dst;
   transfer_host_barrier.size = image_buffer_size;
   CmdPipelineBarrier(image_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                      &transfer_host_barrier, 0, nullptr);
   result = EndCommandBuffer(image_command);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkEndCommandBuffer", result);
   submit.pCommandBuffers = &image_command;
   result = QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkQueueSubmit", result);
   result = QueueWaitIdle(queue);
   if (result != VK_SUCCESS)
      return failed("image-transfer vkQueueWaitIdle", result);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = InvalidateMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("image-transfer vkInvalidateMappedMemoryRanges", result);
   }

   uint64_t image_expected_hash = 1469598103934665603ull;
   uint64_t image_actual_hash = 1469598103934665603ull;
   for (VkDeviceSize i = 0; i < image_buffer_size; ++i) {
      image_expected_hash =
         (image_expected_hash ^ transfer_expected[i]) * 1099511628211ull;
      image_actual_hash =
         (image_actual_hash ^ actual_bytes[i]) * 1099511628211ull;
   }
   if (std::memcmp(transfer_expected.data(), dst_map, image_buffer_size) != 0) {
      VkDeviceSize first = 0;
      while (first < image_buffer_size &&
             transfer_expected[first] == actual_bytes[first])
         ++first;
      LOGE("image-transfer[%s]: FAIL byte=%llu expected=%02x actual=%02x "
           "expected_hash=%016llx actual_hash=%016llx",
           label, static_cast<unsigned long long>(first),
           transfer_expected[first], actual_bytes[first],
           static_cast<unsigned long long>(image_expected_hash),
           static_cast<unsigned long long>(image_actual_hash));
      cleanup();
      return false;
   }
   LOGI("image-transfer[%s]: PASS extent=31x29 rowLength=32 imageHeight=30 "
        "layers=2 hash=%016llx",
        label,
        static_cast<unsigned long long>(image_actual_hash));

   VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
   view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
   view_info.format = image_format;
   view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   view_info.subresourceRange.levelCount = 1;
   view_info.subresourceRange.baseArrayLayer = 1;
   view_info.subresourceRange.layerCount = 1;
   view_info.image = sampled_image;
   result = CreateImageView(device, &view_info, nullptr, &sampled_image_view);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateImageView(source)", result);
   view_info.image = output_image;
   view_info.subresourceRange.baseArrayLayer = 0;
   result = CreateImageView(device, &view_info, nullptr, &output_image_view);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateImageView(output)", result);

   VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
   sampler_info.magFilter = VK_FILTER_NEAREST;
   sampler_info.minFilter = VK_FILTER_NEAREST;
   sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
   sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sampler_info.maxLod = 0.0f;
   result = CreateSampler(device, &sampler_info, nullptr, &sampler);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateSampler", result);

   VkDescriptorSetLayoutBinding sample_binding{};
   sample_binding.binding = 0;
   sample_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   sample_binding.descriptorCount = 1;
   sample_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   descriptor_layout_info.pBindings = &sample_binding;
   result = CreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr,
                                      &sample_descriptor_layout);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateDescriptorSetLayout", result);
   VkDescriptorPoolSize sample_pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
   descriptor_pool_info.pPoolSizes = &sample_pool_size;
   result = CreateDescriptorPool(device, &descriptor_pool_info, nullptr,
                                 &sample_descriptor_pool);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateDescriptorPool", result);
   VkDescriptorSet sample_descriptor_set{};
   descriptor_allocate_info.descriptorPool = sample_descriptor_pool;
   descriptor_allocate_info.pSetLayouts = &sample_descriptor_layout;
   result = AllocateDescriptorSets(device, &descriptor_allocate_info,
                                   &sample_descriptor_set);
   if (result != VK_SUCCESS)
      return failed("sample vkAllocateDescriptorSets", result);
   VkDescriptorImageInfo descriptor_image{};
   descriptor_image.sampler = sampler;
   descriptor_image.imageView = sampled_image_view;
   descriptor_image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   descriptor_write.dstSet = sample_descriptor_set;
   descriptor_write.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   descriptor_write.pBufferInfo = nullptr;
   descriptor_write.pImageInfo = &descriptor_image;
   UpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);

   shader_info.codeSize = sizeof(panvk_sample_vert_spv);
   shader_info.pCode = panvk_sample_vert_spv;
   result = CreateShaderModule(device, &shader_info, nullptr,
                               &sample_vertex_shader);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateShaderModule(vertex)", result);
   shader_info.codeSize = sizeof(panvk_sample_frag_spv);
   shader_info.pCode = panvk_sample_frag_spv;
   result = CreateShaderModule(device, &shader_info, nullptr,
                               &sample_fragment_shader);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateShaderModule(fragment)", result);
   pipeline_layout_info.pSetLayouts = &sample_descriptor_layout;
   result = CreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                 &sample_pipeline_layout);
   if (result != VK_SUCCESS)
      return failed("sample vkCreatePipelineLayout", result);

   VkAttachmentDescription color_attachment{};
   color_attachment.format = image_format;
   color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
   color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   color_attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   VkAttachmentReference color_reference{0,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription subpass{};
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   subpass.colorAttachmentCount = 1;
   subpass.pColorAttachments = &color_reference;
   VkRenderPassCreateInfo render_pass_info{
      VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
   render_pass_info.attachmentCount = 1;
   render_pass_info.pAttachments = &color_attachment;
   render_pass_info.subpassCount = 1;
   render_pass_info.pSubpasses = &subpass;
   result = CreateRenderPass(device, &render_pass_info, nullptr,
                             &sample_render_pass);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateRenderPass", result);
   VkFramebufferCreateInfo framebuffer_info{
      VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
   framebuffer_info.renderPass = sample_render_pass;
   framebuffer_info.attachmentCount = 1;
   framebuffer_info.pAttachments = &output_image_view;
   framebuffer_info.width = image_width;
   framebuffer_info.height = image_height;
   framebuffer_info.layers = 1;
   result = CreateFramebuffer(device, &framebuffer_info, nullptr,
                              &sample_framebuffer);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateFramebuffer", result);

   VkPipelineShaderStageCreateInfo sample_stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_VERTEX_BIT, sample_vertex_shader, "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, sample_fragment_shader, "main", nullptr},
   };
   VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
   input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   VkViewport viewport{0.0f, 0.0f, static_cast<float>(image_width),
                       static_cast<float>(image_height), 0.0f, 1.0f};
   VkRect2D scissor{{0, 0}, {image_width, image_height}};
   VkPipelineViewportStateCreateInfo viewport_state{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
   viewport_state.viewportCount = 1;
   viewport_state.pViewports = &viewport;
   viewport_state.scissorCount = 1;
   viewport_state.pScissors = &scissor;
   VkPipelineRasterizationStateCreateInfo rasterization{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rasterization.polygonMode = VK_POLYGON_MODE_FILL;
   rasterization.cullMode = VK_CULL_MODE_NONE;
   rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rasterization.lineWidth = 1.0f;
   VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState blend_attachment{};
   blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   blend.attachmentCount = 1;
   blend.pAttachments = &blend_attachment;
   VkGraphicsPipelineCreateInfo graphics_info{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
   graphics_info.stageCount = 2;
   graphics_info.pStages = sample_stages;
   graphics_info.pVertexInputState = &vertex_input;
   graphics_info.pInputAssemblyState = &input_assembly;
   graphics_info.pViewportState = &viewport_state;
   graphics_info.pRasterizationState = &rasterization;
   graphics_info.pMultisampleState = &multisample;
   graphics_info.pColorBlendState = &blend;
   graphics_info.layout = sample_pipeline_layout;
   graphics_info.renderPass = sample_render_pass;
   result = CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_info,
                                    nullptr, &sample_pipeline);
   if (result != VK_SUCCESS)
      return failed("sample vkCreateGraphicsPipelines", result);

   VkCommandBuffer sample_command{};
   result = AllocateCommandBuffers(device, &command_info, &sample_command);
   if (result != VK_SUCCESS)
      return failed("sample vkAllocateCommandBuffers", result);
   constexpr uint8_t sample_sentinel = 0x3c;
   std::array<uint8_t, image_buffer_size> sample_expected{};
   sample_expected.fill(sample_sentinel);
   for (uint32_t y = 0; y < image_height; ++y) {
      for (uint32_t x = 0; x < image_width; ++x) {
         const size_t source_offset = layer_pitch + y * row_pitch + x * 4;
         const size_t output_offset = y * row_pitch + x * 4;
         std::memcpy(sample_expected.data() + output_offset,
                     upload.data() + source_offset, 4);
      }
   }
   std::memset(dst_map, sample_sentinel, image_buffer_size);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = FlushMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("sample vkFlushMappedMemoryRanges", result);
   }
   result = BeginCommandBuffer(sample_command, &begin_info);
   if (result != VK_SUCCESS)
      return failed("sample vkBeginCommandBuffer", result);
   VkImageMemoryBarrier sample_barriers[2] = {
      {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT,
       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
       VK_QUEUE_FAMILY_IGNORED, sampled_image,
       {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1}},
      {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
       VK_QUEUE_FAMILY_IGNORED, output_image,
       {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
   };
   CmdPipelineBarrier(sample_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      0, 0, nullptr, 0, nullptr, 2, sample_barriers);
   VkRenderPassBeginInfo render_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
   render_begin.renderPass = sample_render_pass;
   render_begin.framebuffer = sample_framebuffer;
   render_begin.renderArea = scissor;
   CmdBeginRenderPass(sample_command, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
   CmdBindPipeline(sample_command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                   sample_pipeline);
   CmdBindDescriptorSets(sample_command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                         sample_pipeline_layout, 0, 1, &sample_descriptor_set,
                         0, nullptr);
   CmdDraw(sample_command, 3, 1, 0, 0);
   CmdEndRenderPass(sample_command);
   VkImageMemoryBarrier output_barrier{
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
   output_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   output_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   output_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   output_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   output_barrier.image = output_image;
   output_barrier.subresourceRange =
      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
   CmdPipelineBarrier(sample_command,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                      nullptr, 1, &output_barrier);
   VkBufferImageCopy sample_copy{};
   sample_copy.bufferRowLength = buffer_row_length;
   sample_copy.bufferImageHeight = buffer_image_height;
   sample_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   sample_copy.imageSubresource.baseArrayLayer = 0;
   sample_copy.imageSubresource.layerCount = 1;
   sample_copy.imageExtent = {image_width, image_height, 1};
   CmdCopyImageToBuffer(sample_command, output_image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1,
                        &sample_copy);
   CmdPipelineBarrier(sample_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                      &transfer_host_barrier, 0, nullptr);
   result = EndCommandBuffer(sample_command);
   if (result != VK_SUCCESS)
      return failed("sample vkEndCommandBuffer", result);
   submit.pCommandBuffers = &sample_command;
   result = QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
   if (result != VK_SUCCESS)
      return failed("sample vkQueueSubmit", result);
   result = QueueWaitIdle(queue);
   if (result != VK_SUCCESS)
      return failed("sample vkQueueWaitIdle", result);
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = dst_memory;
      range.size = VK_WHOLE_SIZE;
      result = InvalidateMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("sample vkInvalidateMappedMemoryRanges", result);
   }

   uint64_t sample_expected_hash = 1469598103934665603ull;
   uint64_t sample_actual_hash = 1469598103934665603ull;
   for (VkDeviceSize i = 0; i < image_buffer_size; ++i) {
      sample_expected_hash =
         (sample_expected_hash ^ sample_expected[i]) * 1099511628211ull;
      sample_actual_hash =
         (sample_actual_hash ^ actual_bytes[i]) * 1099511628211ull;
   }
   if (std::memcmp(sample_expected.data(), dst_map, image_buffer_size) != 0) {
      VkDeviceSize first = 0;
      while (first < image_buffer_size &&
             sample_expected[first] == actual_bytes[first])
         ++first;
      LOGE("sample[%s]: FAIL byte=%llu expected=%02x actual=%02x "
           "expected_hash=%016llx actual_hash=%016llx",
           label, static_cast<unsigned long long>(first),
           sample_expected[first], actual_bytes[first],
           static_cast<unsigned long long>(sample_expected_hash),
           static_cast<unsigned long long>(sample_actual_hash));
      cleanup();
      return false;
   }
   LOGI("sample[%s]: PASS extent=31x29 rowLength=32 imageHeight=30 "
        "sourceLayer=1 hash=%016llx",
        label,
        static_cast<unsigned long long>(sample_actual_hash));

   VkFormat depth_format = VK_FORMAT_UNDEFINED;
   VkImageFormatProperties depth_properties{};
   result = GetPhysicalDeviceImageFormatProperties(
      physical, VK_FORMAT_D16_UNORM, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0,
      &depth_properties);
   if (result != VK_SUCCESS && result != VK_ERROR_FORMAT_NOT_SUPPORTED)
      return failed("draw D16 vkGetPhysicalDeviceImageFormatProperties",
                    result);
   if (result == VK_SUCCESS &&
       depth_properties.maxExtent.width >= image_width &&
       depth_properties.maxExtent.height >= image_height &&
       (depth_properties.sampleCounts & VK_SAMPLE_COUNT_1_BIT)) {
      depth_format = VK_FORMAT_D16_UNORM;
   } else {
      result = GetPhysicalDeviceImageFormatProperties(
         physical, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D,
         VK_IMAGE_TILING_OPTIMAL,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0, &depth_properties);
      if (result != VK_SUCCESS && result != VK_ERROR_FORMAT_NOT_SUPPORTED)
         return failed("draw D32 vkGetPhysicalDeviceImageFormatProperties",
                       result);
      if (result == VK_SUCCESS &&
          depth_properties.maxExtent.width >= image_width &&
          depth_properties.maxExtent.height >= image_height &&
          (depth_properties.sampleCounts & VK_SAMPLE_COUNT_1_BIT))
         depth_format = VK_FORMAT_D32_SFLOAT;
   }
   if (depth_format == VK_FORMAT_UNDEFINED) {
      LOGI("draw[%s]: SKIP unsupported D16/D32 depth attachment", label);
      cleanup();
      return false;
   }

   image_info.format = depth_format;
   image_info.arrayLayers = 1;
   image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   result = CreateImage(device, &image_info, nullptr, &depth_image);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateImage(depth)", result);
   if (!allocate_image_memory(depth_image, &depth_image_memory, "draw-depth")) {
      cleanup();
      return false;
   }
   view_info.image = depth_image;
   view_info.format = depth_format;
   view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   view_info.subresourceRange.baseArrayLayer = 0;
   view_info.subresourceRange.layerCount = 1;
   result = CreateImageView(device, &view_info, nullptr, &depth_image_view);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateImageView(depth)", result);

   shader_info.codeSize = sizeof(panvk_draw_vert_spv);
   shader_info.pCode = panvk_draw_vert_spv;
   result = CreateShaderModule(device, &shader_info, nullptr,
                               &draw_vertex_shader);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateShaderModule(vertex)", result);
   shader_info.codeSize = sizeof(panvk_draw_frag_spv);
   shader_info.pCode = panvk_draw_frag_spv;
   result = CreateShaderModule(device, &shader_info, nullptr,
                               &draw_fragment_shader);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateShaderModule(fragment)", result);

   VkPushConstantRange push_range{};
   push_range.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
   push_range.size = 32;
   VkPipelineLayoutCreateInfo draw_layout_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   draw_layout_info.pushConstantRangeCount = 1;
   draw_layout_info.pPushConstantRanges = &push_range;
   result = CreatePipelineLayout(device, &draw_layout_info, nullptr,
                                 &draw_pipeline_layout);
   if (result != VK_SUCCESS)
      return failed("draw vkCreatePipelineLayout", result);

   VkAttachmentDescription draw_attachments[2]{};
   draw_attachments[0].format = image_format;
   draw_attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
   draw_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   draw_attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   draw_attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   draw_attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   draw_attachments[0].initialLayout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   draw_attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   draw_attachments[1].format = depth_format;
   draw_attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
   draw_attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   draw_attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   draw_attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   draw_attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   draw_attachments[1].initialLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   draw_attachments[1].finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   VkAttachmentReference draw_color_reference{
      0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkAttachmentReference draw_depth_reference{
      1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   VkSubpassDescription draw_subpass{};
   draw_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   draw_subpass.colorAttachmentCount = 1;
   draw_subpass.pColorAttachments = &draw_color_reference;
   draw_subpass.pDepthStencilAttachment = &draw_depth_reference;
   VkRenderPassCreateInfo draw_render_pass_info{
      VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
   draw_render_pass_info.attachmentCount = 2;
   draw_render_pass_info.pAttachments = draw_attachments;
   draw_render_pass_info.subpassCount = 1;
   draw_render_pass_info.pSubpasses = &draw_subpass;
   result = CreateRenderPass(device, &draw_render_pass_info, nullptr,
                             &draw_render_pass);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateRenderPass", result);
   VkImageView draw_views[2] = {output_image_view, depth_image_view};
   framebuffer_info.renderPass = draw_render_pass;
   framebuffer_info.attachmentCount = 2;
   framebuffer_info.pAttachments = draw_views;
   result = CreateFramebuffer(device, &framebuffer_info, nullptr,
                              &draw_framebuffer);
   if (result != VK_SUCCESS)
      return failed("draw vkCreateFramebuffer", result);

   VkPipelineShaderStageCreateInfo draw_stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_VERTEX_BIT, draw_vertex_shader, "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, draw_fragment_shader, "main", nullptr},
   };
   VkPipelineDepthStencilStateCreateInfo depth_state{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
   depth_state.depthCompareOp = VK_COMPARE_OP_ALWAYS;
   VkPipelineColorBlendAttachmentState draw_blend_attachment{};
   draw_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   VkPipelineColorBlendStateCreateInfo draw_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   draw_blend.attachmentCount = 1;
   draw_blend.pAttachments = &draw_blend_attachment;
   graphics_info.pStages = draw_stages;
   graphics_info.pDepthStencilState = &depth_state;
   graphics_info.pColorBlendState = &draw_blend;
   graphics_info.layout = draw_pipeline_layout;
   graphics_info.renderPass = draw_render_pass;
   rasterization.cullMode = VK_CULL_MODE_BACK_BIT;
   result = CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_info,
                                    nullptr, &indexed_pipeline);
   if (result != VK_SUCCESS)
      return failed("indexed vkCreateGraphicsPipelines", result);

   draw_blend_attachment.blendEnable = VK_TRUE;
   rasterization.cullMode = VK_CULL_MODE_NONE;
   draw_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
   draw_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
   draw_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
   draw_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   draw_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
   draw_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
   result = CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_info,
                                    nullptr, &blend_pipeline);
   if (result != VK_SUCCESS)
      return failed("blend vkCreateGraphicsPipelines", result);

   draw_blend_attachment.blendEnable = VK_FALSE;
   depth_state.depthTestEnable = VK_TRUE;
   depth_state.depthWriteEnable = VK_TRUE;
   depth_state.depthCompareOp = VK_COMPARE_OP_LESS;
   result = CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_info,
                                    nullptr, &depth_pipeline);
   if (result != VK_SUCCESS)
      return failed("depth vkCreateGraphicsPipelines", result);

   constexpr VkDeviceSize index_offset = image_buffer_size;
   const uint16_t indices[3] = {0, 2, 1};
   std::memcpy(static_cast<uint8_t *>(src_map) + index_offset, indices,
               sizeof(indices));
   if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = src_memory;
      range.size = VK_WHOLE_SIZE;
      result = FlushMappedMemoryRanges(device, 1, &range);
      if (result != VK_SUCCESS)
         return failed("indexed vkFlushMappedMemoryRanges", result);
   }

   struct DrawPush {
      float color[4];
      float params[4];
   };
   static_assert(sizeof(DrawPush) == 32, "unexpected draw push size");
   auto run_draw_probe = [&](const char *probe, VkPipeline pipeline,
                             uint8_t sentinel,
                             const std::array<uint8_t, 4> &expected_pixel,
                             uint32_t mode) {
      std::array<uint8_t, image_buffer_size> draw_expected{};
      draw_expected.fill(sentinel);
      for (uint32_t y = 0; y < image_height; ++y) {
         for (uint32_t x = 0; x < image_width; ++x) {
            const size_t offset = y * row_pitch + x * 4;
            std::memcpy(draw_expected.data() + offset, expected_pixel.data(),
                        expected_pixel.size());
         }
      }
      std::memset(dst_map, sentinel, image_buffer_size);
      if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = dst_memory;
         range.size = VK_WHOLE_SIZE;
         VkResult flush_result = FlushMappedMemoryRanges(device, 1, &range);
         if (flush_result != VK_SUCCESS) {
            LOGE("%s[%s]: vkFlushMappedMemoryRanges failed: %d", probe, label,
                 flush_result);
            return false;
         }
      }

      VkCommandBuffer draw_command{};
      VkResult draw_result =
         AllocateCommandBuffers(device, &command_info, &draw_command);
      if (draw_result != VK_SUCCESS) {
         LOGE("%s[%s]: vkAllocateCommandBuffers failed: %d", probe, label,
              draw_result);
         return false;
      }
      draw_result = BeginCommandBuffer(draw_command, &begin_info);
      if (draw_result != VK_SUCCESS) {
         LOGE("%s[%s]: vkBeginCommandBuffer failed: %d", probe, label,
              draw_result);
         return false;
      }
      VkImageMemoryBarrier draw_color_barrier{
         VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      draw_color_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      draw_color_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      draw_color_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      draw_color_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      draw_color_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      draw_color_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      draw_color_barrier.image = output_image;
      draw_color_barrier.subresourceRange =
         {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      CmdPipelineBarrier(draw_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &draw_color_barrier);
      if (mode == 0) {
         VkImageMemoryBarrier depth_barrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
         depth_barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
         depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
         depth_barrier.newLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
         depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         depth_barrier.image = depth_image;
         depth_barrier.subresourceRange =
            {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
         CmdPipelineBarrier(draw_command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0,
                            nullptr, 0, nullptr, 1, &depth_barrier);
         VkBufferMemoryBarrier index_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
         index_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
         index_barrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
         index_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         index_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
         index_barrier.buffer = src;
         index_barrier.offset = index_offset;
         index_barrier.size = sizeof(indices);
         CmdPipelineBarrier(draw_command, VK_PIPELINE_STAGE_HOST_BIT,
                            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr,
                            1, &index_barrier, 0, nullptr);
      }

      VkClearValue clears[2]{};
      if (mode == 1)
         clears[0].color.float32[0] = 1.0f;
      clears[1].depthStencil.depth = 1.0f;
      VkRenderPassBeginInfo draw_begin{
         VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      draw_begin.renderPass = draw_render_pass;
      draw_begin.framebuffer = draw_framebuffer;
      draw_begin.renderArea = scissor;
      draw_begin.clearValueCount = 2;
      draw_begin.pClearValues = clears;
      CmdBeginRenderPass(draw_command, &draw_begin, VK_SUBPASS_CONTENTS_INLINE);
      CmdBindPipeline(draw_command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      DrawPush push{};
      if (mode == 0) {
         push.color[2] = push.color[3] = 1.0f;
         push.params[0] = 0.5f;
         CmdPushConstants(draw_command, draw_pipeline_layout,
                          VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                          0, sizeof(push), &push);
         CmdBindIndexBuffer(draw_command, src, index_offset,
                            VK_INDEX_TYPE_UINT16);
         CmdDrawIndexed(draw_command, 3, 1, 0, 0, 0);
      } else if (mode == 1) {
         push.color[1] = push.color[3] = 1.0f;
         push.params[0] = 0.5f;
         CmdPushConstants(draw_command, draw_pipeline_layout,
                          VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                          0, sizeof(push), &push);
         CmdDraw(draw_command, 3, 1, 0, 0);
      } else {
         push.color[1] = push.color[3] = 1.0f;
         push.params[0] = 0.25f;
         CmdPushConstants(draw_command, draw_pipeline_layout,
                          VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                          0, sizeof(push), &push);
         CmdDraw(draw_command, 3, 1, 0, 0);
         std::memset(&push, 0, sizeof(push));
         push.color[0] = push.color[3] = 1.0f;
         push.params[0] = 0.75f;
         CmdPushConstants(draw_command, draw_pipeline_layout,
                          VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                          0, sizeof(push), &push);
         CmdDraw(draw_command, 3, 1, 0, 0);
      }
      CmdEndRenderPass(draw_command);
      draw_color_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      draw_color_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      draw_color_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      draw_color_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      CmdPipelineBarrier(draw_command,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &draw_color_barrier);
      CmdCopyImageToBuffer(draw_command, output_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1,
                           &sample_copy);
      CmdPipelineBarrier(draw_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &transfer_host_barrier, 0, nullptr);
      draw_result = EndCommandBuffer(draw_command);
      if (draw_result != VK_SUCCESS) {
         LOGE("%s[%s]: vkEndCommandBuffer failed: %d", probe, label,
              draw_result);
         return false;
      }
      VkSubmitInfo draw_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      draw_submit.commandBufferCount = 1;
      draw_submit.pCommandBuffers = &draw_command;
      draw_result = QueueSubmit(queue, 1, &draw_submit, VK_NULL_HANDLE);
      if (draw_result == VK_SUCCESS)
         draw_result = QueueWaitIdle(queue);
      if (draw_result != VK_SUCCESS) {
         LOGE("%s[%s]: submit/wait failed: %d", probe, label, draw_result);
         return false;
      }
      if (!(memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
         range.memory = dst_memory;
         range.size = VK_WHOLE_SIZE;
         draw_result = InvalidateMappedMemoryRanges(device, 1, &range);
         if (draw_result != VK_SUCCESS) {
            LOGE("%s[%s]: vkInvalidateMappedMemoryRanges failed: %d", probe,
                 label, draw_result);
            return false;
         }
      }
      uint64_t expected_hash = 1469598103934665603ull;
      uint64_t actual_hash = 1469598103934665603ull;
      for (VkDeviceSize i = 0; i < image_buffer_size; ++i) {
         expected_hash =
            (expected_hash ^ draw_expected[i]) * 1099511628211ull;
         actual_hash =
            (actual_hash ^ actual_bytes[i]) * 1099511628211ull;
      }
      if (std::memcmp(draw_expected.data(), dst_map, image_buffer_size) != 0) {
         VkDeviceSize first = 0;
         while (first < image_buffer_size &&
                draw_expected[first] == actual_bytes[first])
            ++first;
         LOGE("%s[%s]: FAIL byte=%llu expected=%02x actual=%02x "
              "expected_hash=%016llx actual_hash=%016llx",
              probe, label, static_cast<unsigned long long>(first),
              draw_expected[first], actual_bytes[first],
              static_cast<unsigned long long>(expected_hash),
              static_cast<unsigned long long>(actual_hash));
         return false;
      }
      LOGI("%s[%s]: PASS extent=31x29 rowLength=32 imageHeight=30 "
           "hash=%016llx",
           probe, label, static_cast<unsigned long long>(actual_hash));
      return true;
   };

   if (!run_draw_probe("indexed", indexed_pipeline, 0x27,
                       {0, 0, 255, 255}, 0) ||
       !run_draw_probe("blend", blend_pipeline, 0x39,
                       {255, 255, 0, 255}, 1) ||
       !run_draw_probe("depth", depth_pipeline, 0x4b,
                       {0, 255, 0, 255}, 2)) {
      cleanup();
      return false;
   }
   cleanup();
   return true;
}

bool panvk_present_init(void *libvulkan, ANativeWindow *window)
{
   reset_hud("Initializing");
   if (!libvulkan || !window) {
      reset_hud("Unavailable");
      return false;
   }

   s.loader = libvulkan;
   s.GetInstanceProcAddr = loader_proc<PFN_vkGetInstanceProcAddr>(
      "vkGetInstanceProcAddr");
   auto CreateInstance = loader_proc<PFN_vkCreateInstance>("vkCreateInstance");
   if (!s.GetInstanceProcAddr || !CreateInstance) {
      LOGE("Vulkan loader entrypoints unavailable");
      return false;
   }

   const char *instance_extensions[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
   };
   VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
   app_info.pApplicationName = "PanVK Android present probe";
   app_info.apiVersion = VK_API_VERSION_1_0;
   VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   instance_info.pApplicationInfo = &app_info;
   instance_info.enabledExtensionCount = 2;
   instance_info.ppEnabledExtensionNames = instance_extensions;
   if (!check(CreateInstance(&instance_info, nullptr, &s.instance),
              "vkCreateInstance"))
      return false;

   s.DestroyInstance = instance_proc<PFN_vkDestroyInstance>("vkDestroyInstance");
   auto EnumeratePhysicalDevices = instance_proc<PFN_vkEnumeratePhysicalDevices>(
      "vkEnumeratePhysicalDevices");
   auto GetPhysicalDeviceProperties =
      instance_proc<PFN_vkGetPhysicalDeviceProperties>(
         "vkGetPhysicalDeviceProperties");
   auto GetQueueFamilies = instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      "vkGetPhysicalDeviceQueueFamilyProperties");
   auto GetSurfaceSupport = loader_proc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
      "vkGetPhysicalDeviceSurfaceSupportKHR");
   auto GetSurfaceCapabilities = loader_proc<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
      "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
   auto GetSurfaceFormats = loader_proc<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
      "vkGetPhysicalDeviceSurfaceFormatsKHR");
   auto CreateAndroidSurface = loader_proc<PFN_vkCreateAndroidSurfaceKHR>(
      "vkCreateAndroidSurfaceKHR");
   auto CreateDevice = instance_proc<PFN_vkCreateDevice>("vkCreateDevice");
   s.DestroySurfaceKHR = loader_proc<PFN_vkDestroySurfaceKHR>("vkDestroySurfaceKHR");
   if (!EnumeratePhysicalDevices || !GetPhysicalDeviceProperties ||
       !GetQueueFamilies || !GetSurfaceSupport || !GetSurfaceCapabilities ||
       !GetSurfaceFormats || !CreateAndroidSurface || !CreateDevice ||
       !s.DestroySurfaceKHR) {
      LOGE("required instance entrypoint unavailable");
      panvk_present_shutdown();
      return false;
   }

   VkAndroidSurfaceCreateInfoKHR surface_info{
      VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
   surface_info.window = window;
   if (!check(CreateAndroidSurface(s.instance, &surface_info, nullptr, &s.surface),
              "vkCreateAndroidSurfaceKHR")) {
      panvk_present_shutdown();
      return false;
   }

   uint32_t physical_count = 1;
   if (!check(EnumeratePhysicalDevices(s.instance, &physical_count, &s.physical),
              "vkEnumeratePhysicalDevices") || physical_count == 0) {
      panvk_present_shutdown();
      return false;
   }
   VkPhysicalDeviceProperties physical_properties{};
   GetPhysicalDeviceProperties(s.physical, &physical_properties);
   s.device_name = physical_properties.deviceName;

   uint32_t family_count = 0;
   GetQueueFamilies(s.physical, &family_count, nullptr);
   std::vector<VkQueueFamilyProperties> families(family_count);
   GetQueueFamilies(s.physical, &family_count, families.data());
   bool found_family = false;
   for (uint32_t i = 0; i < family_count; ++i) {
      VkBool32 supports_present = VK_FALSE;
      if (GetSurfaceSupport(s.physical, i, s.surface, &supports_present) == VK_SUCCESS &&
          supports_present && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
         s.queue_family = i;
         found_family = true;
         break;
      }
   }
   if (!found_family) {
      LOGE("no graphics+present queue family");
      panvk_present_shutdown();
      return false;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
   queue_info.queueFamilyIndex = s.queue_family;
   queue_info.queueCount = 1;
   queue_info.pQueuePriorities = &priority;
   const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
   VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
   device_info.queueCreateInfoCount = 1;
   device_info.pQueueCreateInfos = &queue_info;
   device_info.enabledExtensionCount = 1;
   device_info.ppEnabledExtensionNames = device_extensions;
   if (!check(CreateDevice(s.physical, &device_info, nullptr, &s.device),
              "vkCreateDevice")) {
      panvk_present_shutdown();
      return false;
   }

   s.GetDeviceProcAddr = loader_proc<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
   auto GetDeviceQueue = device_proc<PFN_vkGetDeviceQueue>("vkGetDeviceQueue");
   auto CreateSwapchain = loader_proc<PFN_vkCreateSwapchainKHR>("vkCreateSwapchainKHR");
   auto GetSwapchainImages = loader_proc<PFN_vkGetSwapchainImagesKHR>(
      "vkGetSwapchainImagesKHR");
   auto CreateCommandPool = device_proc<PFN_vkCreateCommandPool>("vkCreateCommandPool");
   auto AllocateCommandBuffers = device_proc<PFN_vkAllocateCommandBuffers>(
      "vkAllocateCommandBuffers");
   auto CreateSemaphore = device_proc<PFN_vkCreateSemaphore>("vkCreateSemaphore");
   auto CreateFence = device_proc<PFN_vkCreateFence>("vkCreateFence");
   auto CreateImageView = device_proc<PFN_vkCreateImageView>("vkCreateImageView");
   auto CreateRenderPass = device_proc<PFN_vkCreateRenderPass>("vkCreateRenderPass");
   auto CreateFramebuffer = device_proc<PFN_vkCreateFramebuffer>("vkCreateFramebuffer");
   auto CreateShaderModule = device_proc<PFN_vkCreateShaderModule>("vkCreateShaderModule");
   auto DestroyShaderModule = device_proc<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
   auto CreatePipelineLayout = device_proc<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
   auto CreateGraphicsPipelines = device_proc<PFN_vkCreateGraphicsPipelines>(
      "vkCreateGraphicsPipelines");
   s.DestroyDevice = device_proc<PFN_vkDestroyDevice>("vkDestroyDevice");
   s.DestroySwapchainKHR = loader_proc<PFN_vkDestroySwapchainKHR>("vkDestroySwapchainKHR");
   s.DestroyRenderPass = device_proc<PFN_vkDestroyRenderPass>("vkDestroyRenderPass");
   s.DestroyPipelineLayout = device_proc<PFN_vkDestroyPipelineLayout>(
      "vkDestroyPipelineLayout");
   s.DestroyPipeline = device_proc<PFN_vkDestroyPipeline>("vkDestroyPipeline");
   s.DestroyImageView = device_proc<PFN_vkDestroyImageView>("vkDestroyImageView");
   s.DestroyFramebuffer = device_proc<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer");
   s.DestroyCommandPool = device_proc<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
   s.DestroySemaphore = device_proc<PFN_vkDestroySemaphore>("vkDestroySemaphore");
   s.DestroyFence = device_proc<PFN_vkDestroyFence>("vkDestroyFence");
   s.AcquireNextImageKHR = loader_proc<PFN_vkAcquireNextImageKHR>("vkAcquireNextImageKHR");
   s.ResetFences = device_proc<PFN_vkResetFences>("vkResetFences");
   s.WaitForFences = device_proc<PFN_vkWaitForFences>("vkWaitForFences");
   s.ResetCommandBuffer = device_proc<PFN_vkResetCommandBuffer>("vkResetCommandBuffer");
   s.BeginCommandBuffer = device_proc<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
   s.CmdPipelineBarrier = device_proc<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
   s.CmdBeginRenderPass = device_proc<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass");
   s.CmdEndRenderPass = device_proc<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass");
   s.CmdBindPipeline = device_proc<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
   s.CmdSetViewport = device_proc<PFN_vkCmdSetViewport>("vkCmdSetViewport");
   s.CmdSetScissor = device_proc<PFN_vkCmdSetScissor>("vkCmdSetScissor");
   s.CmdPushConstants =
      device_proc<PFN_vkCmdPushConstants>("vkCmdPushConstants");
   s.CmdDraw = device_proc<PFN_vkCmdDraw>("vkCmdDraw");
   s.EndCommandBuffer = device_proc<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
   s.QueueSubmit = device_proc<PFN_vkQueueSubmit>("vkQueueSubmit");
   s.QueuePresentKHR = loader_proc<PFN_vkQueuePresentKHR>("vkQueuePresentKHR");
   s.DeviceWaitIdle = device_proc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
   if (!s.GetDeviceProcAddr || !GetDeviceQueue || !CreateSwapchain ||
       !GetSwapchainImages || !CreateCommandPool || !AllocateCommandBuffers ||
       !CreateSemaphore || !CreateFence || !CreateImageView ||
       !CreateRenderPass || !CreateFramebuffer || !CreateShaderModule ||
       !DestroyShaderModule || !CreatePipelineLayout ||
       !CreateGraphicsPipelines || !s.DestroyDevice ||
       !s.AcquireNextImageKHR || !s.QueuePresentKHR || !s.CmdBeginRenderPass ||
       !s.CmdEndRenderPass || !s.CmdBindPipeline || !s.CmdSetViewport ||
       !s.CmdSetScissor || !s.CmdPushConstants || !s.CmdDraw) {
      LOGE("required device entrypoint unavailable");
      panvk_present_shutdown();
      return false;
   }
   GetDeviceQueue(s.device, s.queue_family, 0, &s.queue);

   VkSurfaceCapabilitiesKHR caps{};
   if (!check(GetSurfaceCapabilities(s.physical, s.surface, &caps),
              "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) {
      panvk_present_shutdown();
      return false;
   }
   uint32_t format_count = 0;
   GetSurfaceFormats(s.physical, s.surface, &format_count, nullptr);
   std::vector<VkSurfaceFormatKHR> formats(format_count);
   GetSurfaceFormats(s.physical, s.surface, &format_count, formats.data());
   if (formats.empty()) {
      LOGE("surface has no formats");
      panvk_present_shutdown();
      return false;
   }
   VkSurfaceFormatKHR format = formats[0];
   for (const auto &candidate : formats) {
      if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM ||
          candidate.format == VK_FORMAT_B8G8R8A8_UNORM) {
         format = candidate;
         break;
      }
   }
   uint32_t image_count = caps.minImageCount + 1;
   if (caps.maxImageCount)
      image_count = std::min(image_count, caps.maxImageCount);
   s.extent = caps.currentExtent;
   if (s.extent.width == UINT32_MAX) {
      s.extent.width = ANativeWindow_getWidth(window);
      s.extent.height = ANativeWindow_getHeight(window);
   }
   VkSwapchainCreateInfoKHR swapchain_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
   swapchain_info.surface = s.surface;
   swapchain_info.minImageCount = image_count;
   swapchain_info.imageFormat = format.format;
   swapchain_info.imageColorSpace = format.colorSpace;
   swapchain_info.imageExtent = s.extent;
   swapchain_info.imageArrayLayers = 1;
   swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
   swapchain_info.preTransform = caps.currentTransform;
   swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
   if (!(caps.supportedCompositeAlpha & swapchain_info.compositeAlpha))
      swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
   swapchain_info.clipped = VK_TRUE;
   if (!check(CreateSwapchain(s.device, &swapchain_info, nullptr, &s.swapchain),
              "vkCreateSwapchainKHR")) {
      panvk_present_shutdown();
      return false;
   }

   uint32_t swapchain_image_count = 0;
   GetSwapchainImages(s.device, s.swapchain, &swapchain_image_count, nullptr);
   s.images.resize(swapchain_image_count);
   GetSwapchainImages(s.device, s.swapchain, &swapchain_image_count, s.images.data());

   VkAttachmentDescription attachment{};
   attachment.format = format.format;
   attachment.samples = VK_SAMPLE_COUNT_1_BIT;
   attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
   VkAttachmentReference color_attachment{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription subpass{};
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   subpass.colorAttachmentCount = 1;
   subpass.pColorAttachments = &color_attachment;
   VkRenderPassCreateInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
   render_pass_info.attachmentCount = 1;
   render_pass_info.pAttachments = &attachment;
   render_pass_info.subpassCount = 1;
   render_pass_info.pSubpasses = &subpass;
   if (!check(CreateRenderPass(s.device, &render_pass_info, nullptr, &s.render_pass),
              "vkCreateRenderPass")) {
      panvk_present_shutdown();
      return false;
   }

   s.image_views.resize(swapchain_image_count);
   s.framebuffers.resize(swapchain_image_count);
   for (uint32_t i = 0; i < swapchain_image_count; ++i) {
      VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view_info.image = s.images[i];
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = format.format;
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.layerCount = 1;
      if (!check(CreateImageView(s.device, &view_info, nullptr, &s.image_views[i]),
                 "vkCreateImageView")) {
         panvk_present_shutdown();
         return false;
      }
      VkFramebufferCreateInfo framebuffer_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      framebuffer_info.renderPass = s.render_pass;
      framebuffer_info.attachmentCount = 1;
      framebuffer_info.pAttachments = &s.image_views[i];
      framebuffer_info.width = s.extent.width;
      framebuffer_info.height = s.extent.height;
      framebuffer_info.layers = 1;
      if (!check(CreateFramebuffer(s.device, &framebuffer_info, nullptr,
                                   &s.framebuffers[i]),
                 "vkCreateFramebuffer")) {
         panvk_present_shutdown();
         return false;
      }
   }

   VkShaderModule vert{};
   VkShaderModule frag{};
   VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
   shader_info.codeSize = sizeof(panvk_triangle_vert_spv);
   shader_info.pCode = panvk_triangle_vert_spv;
   if (!check(CreateShaderModule(s.device, &shader_info, nullptr, &vert),
              "vkCreateShaderModule(vertex)")) {
      panvk_present_shutdown();
      return false;
   }
   shader_info.codeSize = sizeof(panvk_triangle_frag_spv);
   shader_info.pCode = panvk_triangle_frag_spv;
   if (!check(CreateShaderModule(s.device, &shader_info, nullptr, &frag),
              "vkCreateShaderModule(fragment)")) {
      DestroyShaderModule(s.device, vert, nullptr);
      panvk_present_shutdown();
      return false;
   }
   VkPushConstantRange scene_push_range{};
   scene_push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   scene_push_range.size = sizeof(ScenePushConstants);
   VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   layout_info.pushConstantRangeCount = 1;
   layout_info.pPushConstantRanges = &scene_push_range;
   if (!check(CreatePipelineLayout(s.device, &layout_info, nullptr,
                                   &s.pipeline_layout),
              "vkCreatePipelineLayout")) {
      DestroyShaderModule(s.device, frag, nullptr);
      DestroyShaderModule(s.device, vert, nullptr);
      panvk_present_shutdown();
      return false;
   }
   VkPipelineShaderStageCreateInfo stages[2]{};
   stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
   stages[0].module = vert;
   stages[0].pName = "main";
   stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   stages[1].module = frag;
   stages[1].pName = "main";
   VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
   input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   VkPipelineViewportStateCreateInfo viewport_state{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
   viewport_state.viewportCount = 1;
   viewport_state.scissorCount = 1;
   VkPipelineRasterizationStateCreateInfo rasterization{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rasterization.polygonMode = VK_POLYGON_MODE_FILL;
   rasterization.cullMode = VK_CULL_MODE_NONE;
   rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rasterization.lineWidth = 1.0f;
   VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState blend_attachment{};
   blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
   VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   blend.attachmentCount = 1;
   blend.pAttachments = &blend_attachment;
   VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dynamic{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
   dynamic.dynamicStateCount = 2;
   dynamic.pDynamicStates = dynamic_states;
   VkGraphicsPipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
   pipeline_info.stageCount = 2;
   pipeline_info.pStages = stages;
   pipeline_info.pVertexInputState = &vertex_input;
   pipeline_info.pInputAssemblyState = &input_assembly;
   pipeline_info.pViewportState = &viewport_state;
   pipeline_info.pRasterizationState = &rasterization;
   pipeline_info.pMultisampleState = &multisample;
   pipeline_info.pColorBlendState = &blend;
   pipeline_info.pDynamicState = &dynamic;
   pipeline_info.layout = s.pipeline_layout;
   pipeline_info.renderPass = s.render_pass;
   VkResult pipeline_result = CreateGraphicsPipelines(
      s.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &s.pipeline);
   DestroyShaderModule(s.device, frag, nullptr);
   DestroyShaderModule(s.device, vert, nullptr);
   if (!check(pipeline_result, "vkCreateGraphicsPipelines")) {
      panvk_present_shutdown();
      return false;
   }

   VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
   pool_info.queueFamilyIndex = s.queue_family;
   if (!check(CreateCommandPool(s.device, &pool_info, nullptr, &s.pool),
              "vkCreateCommandPool")) {
      panvk_present_shutdown();
      return false;
   }
   VkCommandBufferAllocateInfo command_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
   command_info.commandPool = s.pool;
   command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   command_info.commandBufferCount = 1;
   if (!check(AllocateCommandBuffers(s.device, &command_info, &s.command),
              "vkAllocateCommandBuffers")) {
      panvk_present_shutdown();
      return false;
   }
   VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
   if (!check(CreateSemaphore(s.device, &semaphore_info, nullptr, &s.acquired),
              "vkCreateSemaphore(acquired)") ||
       !check(CreateSemaphore(s.device, &semaphore_info, nullptr, &s.rendered),
              "vkCreateSemaphore(rendered)") ||
       !check(CreateFence(s.device, &fence_info, nullptr, &s.fence),
              "vkCreateFence")) {
      panvk_present_shutdown();
      return false;
   }

   const auto now = std::chrono::steady_clock::now();
   s.scene_start = now;
   s.stats_window_start = now;
   s.last_successful_present = {};
   s.stats_present_calls = 0;
   s.stats_interval_count = 0;
   s.stats_interval_sum_ms = 0.0;
   s.has_successful_present = false;
   publish_hud(0.0f, 0.0, s.extent.width, s.extent.height,
               s.device_name.c_str());
   LOGI("ready extent=%ux%u format=%u images=%u", s.extent.width, s.extent.height,
        format.format, swapchain_image_count);
   return true;
}

void panvk_present_draw()
{
   if (!s.device || !s.swapchain)
      return;
   if (s.WaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
       s.ResetFences(s.device, 1, &s.fence) != VK_SUCCESS)
      return;

   uint32_t image_index = 0;
   VkResult result = s.AcquireNextImageKHR(s.device, s.swapchain, UINT64_MAX,
                                           s.acquired, VK_NULL_HANDLE,
                                           &image_index);
   if (result != VK_SUCCESS)
      return;

   s.ResetCommandBuffer(s.command, 0);
   VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   if (s.BeginCommandBuffer(s.command, &begin) != VK_SUCCESS)
      return;
   VkClearValue clear{};
   clear.color = {{0.02f, 0.035f, 0.065f, 1.0f}};
   VkRenderPassBeginInfo render_pass_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
   render_pass_begin.renderPass = s.render_pass;
   render_pass_begin.framebuffer = s.framebuffers[image_index];
   render_pass_begin.renderArea.extent = s.extent;
   render_pass_begin.clearValueCount = 1;
   render_pass_begin.pClearValues = &clear;
   VkViewport viewport{0.0f, 0.0f, static_cast<float>(s.extent.width),
                       static_cast<float>(s.extent.height), 0.0f, 1.0f};
   VkRect2D scissor{{0, 0}, s.extent};
   s.CmdBeginRenderPass(s.command, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
   s.CmdBindPipeline(s.command, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipeline);
   s.CmdSetViewport(s.command, 0, 1, &viewport);
   s.CmdSetScissor(s.command, 0, 1, &scissor);
   const auto scene_now = std::chrono::steady_clock::now();
   const ScenePushConstants scene_push{
      std::chrono::duration<float>(scene_now - s.scene_start).count(),
      static_cast<float>(s.extent.width),
      static_cast<float>(s.extent.height),
      s.extent.height
         ? static_cast<float>(s.extent.width) /
              static_cast<float>(s.extent.height)
         : 1.0f,
   };
   s.CmdPushConstants(s.command, s.pipeline_layout,
                      VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(scene_push),
                      &scene_push);
   s.CmdDraw(s.command, 3, 1, 0, 0);
   s.CmdEndRenderPass(s.command);
   if (s.EndCommandBuffer(s.command) != VK_SUCCESS)
      return;

   VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
   VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
   submit.waitSemaphoreCount = 1;
   submit.pWaitSemaphores = &s.acquired;
   submit.pWaitDstStageMask = &wait_stage;
   submit.commandBufferCount = 1;
   submit.pCommandBuffers = &s.command;
   submit.signalSemaphoreCount = 1;
   submit.pSignalSemaphores = &s.rendered;
   if (s.QueueSubmit(s.queue, 1, &submit, s.fence) != VK_SUCCESS)
      return;
   VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
   present.waitSemaphoreCount = 1;
   present.pWaitSemaphores = &s.rendered;
   present.swapchainCount = 1;
   present.pSwapchains = &s.swapchain;
   present.pImageIndices = &image_index;
   result = s.QueuePresentKHR(s.queue, &present);
   if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      LOGE("vkQueuePresentKHR failed: %d", result);
      return;
   }

   const auto present_now = std::chrono::steady_clock::now();
   ++s.stats_present_calls;
   if (s.has_successful_present) {
      s.stats_interval_sum_ms +=
         std::chrono::duration<double, std::milli>(
            present_now - s.last_successful_present)
            .count();
      ++s.stats_interval_count;
   }
   s.last_successful_present = present_now;
   s.has_successful_present = true;

   const double window_seconds =
      std::chrono::duration<double>(present_now - s.stats_window_start).count();
   if (window_seconds >= 0.5) {
      const float present_calls_per_second =
         static_cast<float>(s.stats_present_calls / window_seconds);
      const double average_interval_ms = s.stats_interval_count
                                            ? s.stats_interval_sum_ms /
                                                 s.stats_interval_count
                                            : 0.0;
      publish_hud(present_calls_per_second, average_interval_ms,
                  s.extent.width, s.extent.height, s.device_name.c_str());
      s.stats_window_start = present_now;
      s.stats_present_calls = 0;
      s.stats_interval_count = 0;
      s.stats_interval_sum_ms = 0.0;
   }
   if (s.frame++ == 0)
      LOGI("first frame presented");
}

void panvk_present_shutdown()
{
   if (s.device) {
      if (s.DeviceWaitIdle)
         s.DeviceWaitIdle(s.device);
      if (s.fence && s.DestroyFence)
         s.DestroyFence(s.device, s.fence, nullptr);
      if (s.rendered && s.DestroySemaphore)
         s.DestroySemaphore(s.device, s.rendered, nullptr);
      if (s.acquired && s.DestroySemaphore)
         s.DestroySemaphore(s.device, s.acquired, nullptr);
      if (s.pool && s.DestroyCommandPool)
         s.DestroyCommandPool(s.device, s.pool, nullptr);
      if (s.pipeline && s.DestroyPipeline)
         s.DestroyPipeline(s.device, s.pipeline, nullptr);
      if (s.pipeline_layout && s.DestroyPipelineLayout)
         s.DestroyPipelineLayout(s.device, s.pipeline_layout, nullptr);
      if (s.DestroyFramebuffer) {
         for (VkFramebuffer framebuffer : s.framebuffers)
            if (framebuffer)
               s.DestroyFramebuffer(s.device, framebuffer, nullptr);
      }
      if (s.DestroyImageView) {
         for (VkImageView view : s.image_views)
            if (view)
               s.DestroyImageView(s.device, view, nullptr);
      }
      if (s.render_pass && s.DestroyRenderPass)
         s.DestroyRenderPass(s.device, s.render_pass, nullptr);
      if (s.swapchain && s.DestroySwapchainKHR)
         s.DestroySwapchainKHR(s.device, s.swapchain, nullptr);
      if (s.DestroyDevice)
         s.DestroyDevice(s.device, nullptr);
   }
   if (s.surface && s.DestroySurfaceKHR)
      s.DestroySurfaceKHR(s.instance, s.surface, nullptr);
   if (s.instance && s.DestroyInstance)
      s.DestroyInstance(s.instance, nullptr);
   s = {};
   reset_hud("Not active");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_adrenotoolstest2_MainActivity_nativeStats(JNIEnv *environment,
                                                            jobject)
{
   std::string copy;
   {
      std::lock_guard<std::mutex> lock(hud_mutex);
      copy = hud_text;
   }
   return environment->NewStringUTF(copy.c_str());
}
