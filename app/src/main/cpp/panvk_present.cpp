#include "panvk_present.h"

#include <android/log.h>
#include <dlfcn.h>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "panvk_compute_spv.h"
#include "panvk_triangle_frag_spv.h"
#include "panvk_triangle_vert_spv.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PanvkPresent", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PanvkPresent", __VA_ARGS__)

namespace {
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
   PFN_vkCmdDraw CmdDraw{};
   PFN_vkEndCommandBuffer EndCommandBuffer{};
   PFN_vkQueueSubmit QueueSubmit{};
   PFN_vkQueuePresentKHR QueuePresentKHR{};
   PFN_vkDeviceWaitIdle DeviceWaitIdle{};
};

State s;

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
   INSTANCE_PROC(PFN_vkCreateDevice, CreateDevice);
   INSTANCE_PROC(PFN_vkGetDeviceProcAddr, GetDeviceProcAddr);
#undef INSTANCE_PROC

   if (!DestroyInstance || !EnumeratePhysicalDevices ||
       !GetPhysicalDeviceQueueFamilyProperties ||
       !GetPhysicalDeviceMemoryProperties || !CreateDevice ||
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
          (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
         queue_family = i;
         break;
      }
   }
   if (queue_family == UINT32_MAX) {
      LOGE("buffer-copy[%s]: no compute-capable queue", label);
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
   DEVICE_PROC(PFN_vkBeginCommandBuffer, BeginCommandBuffer);
   DEVICE_PROC(PFN_vkCmdCopyBuffer, CmdCopyBuffer);
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
   DEVICE_PROC(PFN_vkDestroyPipeline, DestroyPipeline);
   DEVICE_PROC(PFN_vkCmdBindPipeline, CmdBindPipeline);
   DEVICE_PROC(PFN_vkCmdBindDescriptorSets, CmdBindDescriptorSets);
   DEVICE_PROC(PFN_vkCmdDispatch, CmdDispatch);
   DEVICE_PROC(PFN_vkCmdPipelineBarrier, CmdPipelineBarrier);
   DEVICE_PROC(PFN_vkEndCommandBuffer, EndCommandBuffer);
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
       !BeginCommandBuffer || !CmdCopyBuffer || !CreateDescriptorSetLayout ||
       !DestroyDescriptorSetLayout || !CreateDescriptorPool ||
       !DestroyDescriptorPool || !AllocateDescriptorSets ||
       !UpdateDescriptorSets || !CreateShaderModule || !DestroyShaderModule ||
       !CreatePipelineLayout || !DestroyPipelineLayout ||
       !CreateComputePipelines || !DestroyPipeline || !CmdBindPipeline ||
       !CmdBindDescriptorSets || !CmdDispatch || !CmdPipelineBarrier ||
       !EndCommandBuffer ||
       !QueueSubmit || !QueueWaitIdle || !DeviceWaitIdle) {
      LOGE("buffer-copy[%s]: device entrypoints unavailable", label);
      cleanup();
      return false;
   }

   VkQueue queue{};
   GetDeviceQueue(device, queue_family, 0, &queue);
   VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
   buffer_info.size = size;
   buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
   buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
   cleanup();
   return true;
}

bool panvk_present_init(void *libvulkan, ANativeWindow *window)
{
   if (!libvulkan || !window)
      return false;

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
   if (!EnumeratePhysicalDevices || !GetQueueFamilies || !GetSurfaceSupport ||
       !GetSurfaceCapabilities || !GetSurfaceFormats || !CreateAndroidSurface ||
       !CreateDevice || !s.DestroySurfaceKHR) {
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
       !s.CmdSetScissor || !s.CmdDraw) {
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
   VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
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
   const float phase = static_cast<float>((s.frame++ % 180) / 179.0);
   VkClearValue clear{};
   clear.color = {{0.04f + 0.18f * phase, 0.04f, 0.08f, 1.0f}};
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
   if (result != VK_SUCCESS)
      LOGE("vkQueuePresentKHR failed: %d", result);
   else if (s.frame == 1)
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
}
